import os, sys, time, ljqpy, math, re, json
import unicodedata
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
import numpy as np
import h5py
from functools import partial
from collections import defaultdict
import argparse
from config import config
import random
import copy
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel


import openai
from diskcache import Cache

openai.api_key = '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'
os.environ["OPENAI_API_KEY"] = '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'
openai.proxy = 'http://10.176.40.100:41051'
# openai.proxy = 'http://127.0.0.1:7890'
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
cache = Cache("cache")

parser = argparse.ArgumentParser()
parser.add_argument("--dname", help='dataset for training and testing', choices=['ske2019','HacRED','NYT10-HRL','NYT11-HRL','NYT21-HRL', 'WebNLG', 'WikiKBP', 'NYT10', 'WebNLG_star','CoNLL04'], default='HacRED')
parser.add_argument("--do_train", help='training model for EE model', type=bool, default=False)
parser.add_argument("--do_test", help='test model as a pipeline，output the result as json file', type=bool, default=False)
parser.add_argument("--do_eval", help='evaluate the output file', type=bool, default=False)
parser.add_argument("--cons_candidate", help='construct the candidate pairs for whole test set',type=bool, default=False)
parser.add_argument("--filter", help='whether use triple filter module',type=bool, default=False)
parser.add_argument("--do_predict", help='to predict a single sentence',type=bool, default=False)
# parser.add_argument("--only_llm", help='only use llm for extraction (baseline)',type=bool, default=False)
parser.add_argument("--negative_rate", help='how many ground_truth pair become unlabeled pair.', type=float, default=0)
args = parser.parse_args()
#print(args)
negative_threshold = args.negative_rate
dname = args.dname
datadir = './dataset/' + dname
dsplits = 'train test valid'.split()
fns = {x:os.path.join(datadir, f'new_{x}.json') for x in dsplits}

maxlen = config[dname]['maxlen']
if not os.path.isdir(dname): os.makedirs(dname)
def wdir(x): return os.path.join(dname, x)
rc_threshold = config[dname]['thre_rc']
ee_threshold = config[dname]['thre_ee']

from transformers import BertTokenizer, BertModel, set_seed
set_seed(52)

if dname in ['HacRED', 'ske2019']: plm_name = '/data/dell/dingzepeng/hub/models--hfl--chinese-roberta-wwm-ext/snapshots/5c58d0b8ec1d9014354d691c538661bf00bfdb44'
# if dname in ['HacRED', 'ske2019']: plm_name = 'bert-base-chinese'
else: plm_name = '/data/dell/dingzepeng/hub/models--bert-base-cased/snapshots/5532cc56f74641d4bb33641f5c76a55d11f846e0'
tokenizer = BertTokenizer.from_pretrained(plm_name, model_max_length=maxlen)
with open(os.path.join(datadir, 'rel2id.json')) as fin:
    rel_map = json.load(fin)
rev_rel_map = {v:k for k,v in rel_map.items()}
relation_list = list(rel_map.keys())
rels = None

from utils import TN, restore_token_list, GetTopSpans, FindValuePos

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, negative_threshold):
        self.y = 0
        #global rel
        global rel_map
        #if rels is None:
        #    rels = ljqpy.TokenList(wdir('rels.txt'), 1, data, lambda z:[x['label'] for x in z['relationMentions']], save_low_freq=1)
        print('rels:', len(rel_map))
        #print(rels.t2id)
        self.items = []
        for z in data:
            item = {}
            item['tid'] = torch.tensor(tokenizer.encode(z['sentText'])[:512])
            item['yrc'] = list(set(rel_map[x['label']] for x in z['relationMentions']))
            #item['yrc'] = list(set(rels.t2id[x['label']] for x in z['relationMentions']))
            #item['yrc'] = [i for i in item['yrc'] if random.random() > negative_threshold]
            self.items.append(item)
            self.y += len(item['yrc'])
    def __len__(self): return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k]
        return item['tid'], item['yrc']

class PU_mid_loss(nn.Module):
    def __init__(self, mid=0, pi=0.1):
        super().__init__()
        self.mid = mid
        self.pi = pi

    def forward(self,y_true,y_pred):
        eps = torch.tensor(1e-6).cuda()
        y_true = y_true.double()
        pos = torch.sum(y_true * y_pred, 1) / torch.maximum(eps, torch.sum(y_true, 1))
        pos = - torch.log(pos + eps)
        neg = torch.sum((1-y_true) * y_pred, 1) / torch.maximum(eps, torch.sum(1-y_true, 1))
        neg = torch.abs(neg - self.mid) 
        neg = - torch.log(1 - neg + eps)
        return torch.mean(self.pi*pos + neg)

class DatasetEE(torch.utils.data.Dataset):
    def __init__(self, data, negative_threshold):
        self.items = []
        for i, z in enumerate(data):
            text, spo_list = z['sentText'], z['relationMentions']
            labels = z.get('rc_pred', list(set(x['label'] for x in spo_list)))  
            tokens = tokenizer.tokenize(text)[:maxlen]
            otokens = restore_token_list(text, tokens)
            tid = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
            for label in labels:
                if random.random() <= negative_threshold:
                    continue
                prompt = tokenizer.encode(label)
                plen = len(prompt)
                item = {'text':text, 'spo_list':spo_list}
                item['id'] = i
                item['plen'] = plen
                item['otokens'] = otokens
                item['label'] = label
                item['tid'] = torch.tensor(prompt + tid[1:])
                slen = item['tid'].size(0)
                ss = set(TN(x['em1Text']) for x in spo_list if x['label'] == label)
                oo = set(TN(x['em2Text']) for x in spo_list if x['label'] == label)
                yy = torch.zeros((slen, 4)).float()
                for s in ss:
                    for u, v in FindValuePos(otokens, s): 
                        yy[u+plen,0] = yy[v-1+plen,1] = 1
                for o in oo:
                    for u, v in FindValuePos(otokens, o): 
                        yy[u+plen,2] = yy[v-1+plen,3] = 1
                item['yy'] = yy
                self.items.append(item)
    def __len__(self): 
        return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k%len(self.items)]
        return item['tid'], item['yy']


class DatasetonlyEE(torch.utils.data.Dataset):
    def __init__(self, data, negative_threshold):
        self.items = []
        for i, z in enumerate(data):
            text, spo_list = z['sentText'], z['relationMentions']
            tokens = tokenizer.tokenize(text)[:maxlen]
            otokens = restore_token_list(text, tokens)
            tid = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
            if random.random() <= negative_threshold:
                continue
            item = {'text':text, 'spo_list':spo_list}
            item['id'] = i
            item['otokens'] = otokens
            item['tid'] = torch.tensor(tid[1:])
            slen = item['tid'].size(0)
            ## 标记所有的主体和客体，但是注意，这里并未标记主/客体对，即正负sample是“一致”的（所以还是得过filter）
            ss = set(TN(x['em1Text']) for x in spo_list)
            oo = set(TN(x['em2Text']) for x in spo_list)
            yy = torch.zeros((slen, 4)).float()
            for s in ss:
                for u, v in FindValuePos(otokens, s): 
                    yy[u,0] = yy[v-1,1] = 1
            for o in oo:
                for u, v in FindValuePos(otokens, o): 
                    yy[u,2] = yy[v-1,3] = 1
            item['yy'] = yy
            self.items.append(item)
    def __len__(self): 
        return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k%len(self.items)]
        return item['tid'], item['yy']


def rc_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
    yy = torch.zeros((len(items), len(rel_map)))
    for i, (x, ys) in enumerate(items):
        for y in ys: yy[i,y] = 1
    return xx, yy

def ee_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
    yy = nn.utils.rnn.pad_sequence([y for x,y in items], batch_first=True)
    return xx, yy.float()


class EEModel(nn.Module):
    def __init__(self, outd=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(plm_name)
        self.fc = nn.Linear(768, outd)
    def forward(self, x):
        z = self.bert(x).last_hidden_state
        out = self.fc(z)
        out = torch.sigmoid(out)
        return out

class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    def append(self, out, ans):
        out, ans = set(out), set(ans)
        mid = out & ans
        self.correct += len(mid)
        self.output += len(out)
        self.golden += len(ans)

    def compute(self, show=True):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        if show: print(pstr)
        return f1

    # 为了绘制PR curve打印到文件上
    def compute_and_record(self, fout):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        fout.write(pstr+'\n')
        return (prec, reca, f1)

def tt(t):
    return t['em1Text']+' | '+t['em2Text']

sys.path.append('../')
import pt_utils

# 以下是针对pair的评价，还要写一个最终三元组的评价
def ComputeOne(item, pairs, f1, fout, label=None):
    # spos = [x for x in item['spo_list'] if x['label'] == item['label']]
    spos = [{'em1Text': x['em1Text'], 'em2Text':x['em2Text']} for x in item['spo_list']]
    pairs = set(tt(x) for x in pairs)
    spos = set(tt(x) for x in spos)
    print('-'*30, file=fout)
    print(item['text'], file=fout)
    for x in pairs&spos: print('o', x, file=fout)
    for x in pairs-spos: print('-', x, file=fout)
    for x in spos-pairs: print('+', x, file=fout)
    f1.append(pairs, spos)

def test_ee(): 
    outs = [] 
    with torch.no_grad():
        for x, y in dl_dev:
            out = ee(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    f1 = MetricF1()
    fout = open('ret.txt', 'w', encoding='utf-8')
    for item, rr in zip(dss['test'].items, outs):
        pairs = decode_entitypair(item, rr, ee_threshold)
        ComputeOne(item, pairs, f1, fout)
    f1.compute()
    fout.close()


def decode_entitypair(item, rr, ee_threshold, gpout=None):
    otokens = item['otokens']
    subs = GetTopSpans(otokens, rr[:,:2])
    objs = GetTopSpans(otokens, rr[:,2:])
    vv1 = [x for x,y in subs if y >= 0.1]
    vv2 = [x for x,y in objs if y >= 0.1]
    subv = {x:y for x,y in subs}
    objv = {x:y for x,y in objs}
    pairs = []
    for sv1, sv2 in [(sv1, sv2) for sv1 in vv1 for sv2 in vv2]:
        if gpout is not None:
            loc1, loc2 = FindValuePos(otokens, sv1), FindValuePos(otokens, sv2)
            vals = []
            for u1, v1 in loc1:
                for u2, v2 in loc2:
                    vals.append([])
                    for i in range(1+u1, 1+v1):
                        for j in range(1+u2, 1+v2):
                            vals[-1].append(gpout[0,i,j])
            ind = item['id']
            tdata[ind].setdefault('gp_detail', []).append( (sv1, sv2, vals) )
            vals = [np.array(x).mean() for x in vals]
            tdata[ind].setdefault('gp', []).append( (sv1, sv2, vals) )
            if len(vals) == 0: continue
            if max(vals) < 0:
                continue
        score = min(subv[sv1], objv[sv2])
        if score < ee_threshold: continue
        pairs.append({'em1Text': sv1, 'em2Text':sv2})
    return pairs

# 整个三元组的F1计算（无需decode，因为输出都是大模型搞好的文字）

def tt2(t):
    try:
        ans = t['em1Text']+' | '+t['em2Text']+' | '+t['label']
    except:
        ans = 'wrongcase'
    return ans

def tt2_reverse(t):
    try:
        ans = t['em2Text']+' | '+t['em1Text']+' | '+t['label'] 
    except:
        ans = 'wrongcase'
    return ans   # 为了适应正常语序;兼容主客体如果大模型识别反了的情况

def ComputeOne_2(item, preds, f1, fout, label=None):
    spos = item['std_ans']
    triples = preds['relationMentions']
    triples_1 = [tt2(x) for x in triples]
    triples_reverse = [tt2_reverse(x) for x in triples]
    spos = set(tt2(x) for x in spos)
    triples2 = []
    for i in range(len(triples_1)):
        if triples_1[i] in spos:
            triples2.append(triples_1[i])
        elif triples_reverse[i] in spos:
            triples2.append(triples_reverse[i])
        else:
            triples2.append(triples_1[i])
    triples2 = set(triples2)
    print('-'*30, file=fout)
    print(item['sentText'], file=fout)
    for x in triples2&spos: print('o', x, file=fout)
    for x in triples2-spos: print('-', x, file=fout)
    for x in spos-triples2: print('+', x, file=fout)
    f1.append(triples2, spos)

def test_LLM(filterprompt = False, spo_limit = 1):
    with open(wdir('candidate.json'),'r',encoding='utf-8') as fin:
        outs = json.load(fin)
    if filterprompt:
        path2 = wdir('eval_filter_vicuna_ablation.json')
    else:
        path2 = wdir('eval_origin_vicuna_peft.json')
    with open(path2,'r',encoding='utf-8') as fin:
        preds = json.load(fin)
    f1 = MetricF1()
    fout = open('ret.txt', 'w', encoding='utf-8')
    for i in range(min(len(outs),len(preds))):
        if len(outs[i]['std_ans']) >= spo_limit:
            ComputeOne_2(outs[i], preds[i], f1, fout)
    f1.compute()
    fout.close()

# Train ee
if args.do_train:
    epochs = 25
    dss = {x:DatasetonlyEE(ljqpy.LoadJsons(fn), negative_threshold if x == 'train' else 0) for x, fn in fns.items()}
    dl_train = torch.utils.data.DataLoader(dss['train'], batch_size=16, shuffle=True, collate_fn=ee_collate_fn)
    dl_dev = torch.utils.data.DataLoader(dss['test'], batch_size=16, shuffle=False, collate_fn=ee_collate_fn)
    total_steps = len(dl_train) * epochs

    ee = EEModel().cuda()
    eemfile = wdir(f'ee_negative_{negative_threshold}.pt')
    #ee.load_state_dict(torch.load(eemfile))
    pt_utils.lock_transformer_layers(ee.bert, 3)
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(ee, 5e-5, total_steps)

    loss_fct = lambda y_pred, y_true: - (y_true*torch.log(y_pred+1e-9) + (1-y_true)*torch.log(1-y_pred+1e-9)).mean()
    #loss_fct = lambda y_pred, y_true: - 5*(y_true*torch.log(y_pred+1e-9)).mean() - torch.log(((1-y_true)*(1-y_pred)).mean()+1e-9)

    def train_func(model, ditem):
        x, y = ditem
        y = y.cuda()
        out = model(x.cuda())
        loss = loss_fct(out, y)# + 0.1 * out.mean()
        oc = (out > 0.5).float()
        prec = (oc + y > 1.5).sum() / max(oc.sum().item(), 1)
        reca = (oc + y > 1.5).sum() / max(y.sum().item(), 1)
        f1 = 2 * prec * reca / (prec + reca)
        r = {'loss': loss, 'prec': prec, 'reca': reca, 'f1':f1}
        return r

    pt_utils.train_model(ee, optimizer, dl_train, epochs, train_func, test_ee, 
                   scheduler=scheduler, save_file=eemfile)

if args.cons_candidate:
    tdata = ljqpy.LoadJsons(fns['test'])
    # tdata = ljqpy.LoadJsons(fns['train'])
    ee = EEModel().cuda()
    ee.load_state_dict(torch.load(wdir(f'ee_negative_{negative_threshold}.pt')),strict=False)
    if args.filter:
        import relcomb
        gp = relcomb.GlobalPointerModel(plm_name).cuda()
        gp.load_state_dict(torch.load(wdir(f'relcomb_BCE_sh_{negative_threshold}.pt')),strict=False)
    ds_ee = DatasetonlyEE(tdata, 0)
    dl_ee = torch.utils.data.DataLoader(ds_ee, batch_size=16, shuffle=False, collate_fn=ee_collate_fn)
    gpouts = []
    if args.filter:
        for x, y in dl_ee:
            out = gp(x.cuda()).detach().cpu()
            for z in out: gpouts.append(z.numpy())
        print('gp enabled')
    else:
        gpouts = [None] * len(tdata)
    outs = [] 
    with torch.no_grad():
        for x, y in dl_ee:
            out = ee(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    for item, rr in zip(ds_ee.items, outs):
        triples = decode_entitypair(item, rr, ee_threshold, gpouts[item['id']])
        tdata[item['id']].setdefault('preds', []).extend(triples)
    with open(wdir('candidate.json'), 'w', encoding='utf-8') as fout:
        wdata = [{'sentText':x['sentText'],'preds':x['preds'],'std_ans':x['relationMentions']} for x in tdata]   # 也保留relationMentions便于后续评估
        json.dump(wdata, fout, ensure_ascii=False, indent=2)

if args.do_test:  # 大模型+filter的框架（可以一个流程走，分别收集有提示and没有提示的结果），对应输出两个json供后续评估
    # 记得对不同dataset改一下关系列表
    # inst = '''预定义好下列关系列表：['作者', '毕业院校', '主演', '导演', '制片人', '所属机构', '音乐创作者', '演唱者', '嘉宾', '出版/发行时间', '饰演者', '主持人', '主要人物', '创始人', '成立时间', '首播时间', '妻子', '出版机构', '所处时代', '编剧', '改编自', '女儿', '母亲', '父亲', '法人', '儿子']，请从下面的句子中抽取出包含上述关系的所有三元组。
    # 注意，三元组的关系名称必须从上面的关系列表中选取，不考虑除此以外的其他关系。请按照下面指定的格式进行输出：
    # {"relationMentions": [{"em1Text": 主体1, "em2Text": 客体1, "label": 关系1}, {"em1Text": 主体2, "em2Text": 客体2, "label": 关系2}]}
    # 注意，三元组并不一定只有两个，请你模仿这个格式，将所有符合要求的三元组都输出出来。
    # 下面是一个例子：
    # 输入：丽萨·布伦南·乔布斯，出生于1978年5月17日，美国记者和杂志专栏作家。她是美国苹果公司前CEO史蒂夫·乔布斯与其未婚女友克里斯安·布伦南的女儿，也是乔布斯的长女。
    # 输出：{ "relationMentions": [{"em1Text": "乔布斯", "em2Text": "苹果公司", "label": "所属机构"}, {"em1Text": "乔布斯", "em2Text": "丽萨·布伦南·乔布斯", "label": "女儿"}]}

    # 再次强调，所输出的三元组的关系必须从上面所给的预定义列表中选取，不得输出任何不在列表中的关系。同时，请尽可能多地输出符合要求的三元组。
    # 接下来请模仿这个例子，根据输入，按格式要求输出包含上述关系的所有三元组。注意当实体（主体或客体）可以拆分为两个词语（比如中间有顿号或逗号）时，其应当被拆分为两个三元组而不是合并在一个三元组内。
    # '''
    # inst_english = f'Predefine the following relationship list:{relation_list}, please extract all triples containing the above relationship from the following sentences.'+\
    # '''
    # Note that the relationship name of the triple must be selected from the above relationship list, and other relationships not listed are not considered. Please output according to the specified format below:
    # {"relationMentions": [{"em1Text": subject1, "em2Text": object1, "label": relationship1}, {"em1Text": subject2, "em2Text": object2, "label": relationship2}]}
    # Note that the triple may not only have two, please imitate this format and output all triples that meet the requirements.
    # Here is an example:
    # Input: Pakistan Boxing Federation spokesman Obaid Awan said Chowdhry died of a heart attack in the southern port city of Karachi.
    # Output: {"relationMentions": [{"em1Text": "Chowdhry", "label": "per:country_of_death", "em2Text": "Karachi"}]}

    # Again, it is emphasized that the relationship of the triples output must be selected from the predefined list above, and no relationship not in the list can be output. At the same time, please output as many triples as possible that meet the requirements.
    # Please imitate this example, according to the input, output all triples containing the above relationship according to the format requirements. Note that when the entity (subject or object) can be split into two words (such as a comma or comma in the middle), it should be split into two triples instead of merging into one triple.
    # '''
    inst_english = f'Predefine the following relationship list:{relation_list}, please extract all triples containing the above relationship from the following sentences.'+\
    '''
    Note that the relationship name of the triple must be selected from the above relationship list, and other relationships not listed are not considered. Please output according to the specified format below:
    [{"em1Text": subject1, "em2Text": object1, "label": relationship1}, {"em1Text": subject2, "em2Text": object2, "label": relationship2}]
    Note that the triple may not only have two, please imitate this format and output all triples that meet the requirements.
    Again, it is emphasized that the relationship of the triples output must be selected from the predefined list above, and no relationship not in the list can be output. At the same time, please output as many triples as possible that meet the requirements.
    Please according to the input, output all triples containing the above relationship according to the format requirements. Note that when the entity (subject or object) can be split into two words (such as a comma or comma in the middle), it should be split into two triples instead of merging into one triple.
    '''
    # def chatgpt(wdata):  # 给原句和candidate_pair作为输入，分别保存其初始输出以及经过filter提示后的输出
    #     query_session = [{"role":"user", "content": inst_english +'\n' + wdata['sentText']}]
    #     resp = openai.ChatCompletion.create(
    #         model='gpt-4',
    #         messages=query_session,
    #         temperature=0.8,
    #         max_tokens=2048,
    #         top_p=1,
    #         frequency_penalty=0.0,
    #         presence_penalty=0.0,
    #         request_timeout=60
    #         )
    #     ret_ori = resp.choices[0]['message']
    #     candidates = [(x['em1Text'], x['em2Text']) for x in wdata['preds']]
    #     candi_inst = f'经过检测，上面句子中可能有关系的实体对为{candidates}。请对抽取结果进行检查，并把漏掉的三元组补齐，输出最终结果。'
    #     query_session.append(ret_ori)
    #     query_session.append({"role":"user", "content": candi_inst})
    #     resp2 = openai.ChatCompletion.create(
    #         model='gpt-4',
    #         messages=query_session,
    #         temperature=0.8,
    #         max_tokens=2048,
    #         top_p=1,
    #         frequency_penalty=0.0,
    #         presence_penalty=0.0,
    #         request_timeout=60
    #         )
    #     ret_filter = resp2.choices[0]['message']
    #     return ret_ori["content"], ret_filter["content"]
    # origin_list = []
    # filter_list = []
    # idx = 0
    
    # with open(wdir('candidate.json'), 'r', encoding='utf-8') as fin:
    #     wdatas = json.load(fin)
    #     for wdata in tqdm(wdatas):
    #         try:
    #             time.sleep(45)
    #             p1,p2 = chatgpt(wdata)
    #             origin_list.append(p1)
    #             filter_list.append(p2)
    #         except:
    #             time.sleep(45)
    #             p1,p2 = chatgpt(wdata)
    #             origin_list.append(p1)
    #             filter_list.append(p2)
    #         idx += 1
    #         if idx % 5 == 0:
    #             with open(wdir('origin_llm_gpt4.json'), 'a', encoding='utf-8') as fout:
    #                 json.dump(origin_list, fout, ensure_ascii=False, indent=2)
                
    #             with open(wdir('filter_llm_gpt4.json'), 'a', encoding='utf-8') as fout:
    #                 json.dump(filter_list, fout, ensure_ascii=False, indent=2)
    #             origin_list = []
    #             filter_list = []    
    
    # 用开源模型（如cuteGPT等）
    def generate_prompt(query, history, input=None):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "{}{}\n<end>".format(old_query, response)
        prompt += "{}".format(query)
        return prompt
    model_name = '/data/dell/ljq/llama_13b_112_sft_v1'
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).cuda()
    # model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, '/data/dell/dingzepeng/filter_with_LLM/CuteGPT_finetune/cuteGPT/WikiKBP/WikiKBP_epoch0', fan_in_fan_out=False, low_cpu_mem_usage=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"
    model.eval()
    model = model.to(device)
    def general_llm(wdata): # 给原句和candidate_pair作为输入，分别保存其初始输出以及经过filter提示后的输出
        query_session = inst_english + '\n' + wdata['sentText']
        history = []
        prompt = generate_prompt(query_session, history)
        input_ids = tokenizer(prompt, return_tensors="pt")
        input_ids = input_ids["input_ids"].to(device)
        with torch.no_grad():
            outputs=model.generate(
                    input_ids=input_ids,
                    top_p=0.8,
                    top_k=50,
                    repetition_penalty=1.2,
                    max_new_tokens = 256,
                    early_stopping = True,
                    eos_token_id = tokenizer.convert_tokens_to_ids('<end>'),
                    pad_token_id = tokenizer.eos_token_id,
                    min_length = input_ids.shape[1] + 1
            )
        s = outputs[0]
        # response = tokenizer.decode(s[len(input_ids[0]):])
        response = tokenizer.decode(s)
        response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '')
        history.append((query_session, response))
        
        candidates = [(x['em1Text'], x['em2Text']) for x in wdata['preds']]
        # candi_inst = f'经过检测，上面所给句子中可能有关系的实体对为{candidates}。请对抽取结果进行检查，并把漏掉的三元组补齐，输出最终结果。'
        candi_inst = f'Now we claim that the entity pairs that may be related in the above sentence are {candidates}. Please check the extraction results and fill in the missing triples, remove the wrong triples and output the final result\n'
        prompt = generate_prompt(candi_inst, history)
        input_ids = tokenizer(prompt, return_tensors="pt")
        input_ids = input_ids["input_ids"].to(device)
        with torch.no_grad():
            outputs=model.generate(
                    input_ids=input_ids,
                    top_p=0.8,
                    top_k=50,
                    repetition_penalty=1.2,
                    max_new_tokens = 256,
                    early_stopping = True,
                    eos_token_id = tokenizer.convert_tokens_to_ids('<end>'),
                    pad_token_id = tokenizer.eos_token_id,
                    min_length = input_ids.shape[1] + 1
            )
        s = outputs[0]
        # response2 = tokenizer.decode(s[len(input_ids[0]):])
        response2 = tokenizer.decode(s)
        response2 = response2.replace('<s>', '').replace('<end>', '').replace('</s>', '')
        return response, response2
    
    origin_list = []
    filter_list = []
    idx = 0
    
    with open(wdir('candidate.json'), 'r', encoding='utf-8') as fin:
        wdatas = json.load(fin)
        for wdata in tqdm(wdatas):
            try:
                time.sleep(3)
                p1,p2 = general_llm(wdata)
                origin_list.append(p1)
                filter_list.append(p2)
            except:
                time.sleep(10)
                p1,p2 = general_llm(wdata)
                origin_list.append(p1)
                filter_list.append(p2)
            idx += 1
            if idx % 5 == 0:
                with open(wdir('origin_llm_cutegpt.json'), 'a', encoding='utf-8') as fout:
                    json.dump(origin_list, fout, ensure_ascii=False, indent=2)
                
                with open(wdir('filter_llm_cutegpt.json'), 'a', encoding='utf-8') as fout:
                    json.dump(filter_list, fout, ensure_ascii=False, indent=2)
                origin_list = []
                filter_list = []    


if args.do_predict:
    ee = EEModel().cuda()
    ee.load_state_dict(torch.load(wdir(f'ee_negative_{negative_threshold}.pt')))
    if args.filter:
        import relcomb
        gp = relcomb.GlobalPointerModel(plm_name).cuda()
        gp.load_state_dict(torch.load(wdir(f'relcomb_BCE_sh_{negative_threshold}.pt')))

    while True:
        sent = input('>')
        tdata = [{'sentText': sent, 'relationMentions':[]}]
        
        ds_ee = DatasetonlyEE(tdata, 0)
        dl_ee = torch.utils.data.DataLoader(ds_ee, batch_size=1, shuffle=False, collate_fn=ee_collate_fn)
        gpouts = []
        if args.filter:
            for x, y in dl_ee:
                out = gp(x.cuda()).detach().cpu()
                for z in out: gpouts.append(z.numpy())
            print('gp enabled')
        else:
            gpouts = [None] * len(tdata)
        
        outs = [] 
        with torch.no_grad():
            for x, y in dl_ee:
                out = ee(x.cuda()).detach().cpu()
                for z in out: outs.append(z.numpy())
        for item, rr in zip(ds_ee.items, outs):
            triples = decode_entitypair(item, rr, ee_threshold, gpouts[item['id']])
            tdata[item['id']].setdefault('preds', []).extend(triples)
        for x in tdata[0].get('preds', []):
            print(x)

if args.do_eval:
    for limit in [4,5,7]:
        print(f'limit={limit}')
        print('without filter prompt \n')
        test_LLM(spo_limit=limit)  # 评估大模型结果
        print('with filter prompt\n')
        test_LLM(True, limit)  # 评估大模型+filter结果