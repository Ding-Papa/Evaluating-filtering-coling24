import os, sys, time, ljqpy, math
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
import numpy as np
import h5py
from collections import defaultdict
import random

from transformers import BertTokenizer, BertModel

from utils import TN, restore_token_list, GetTopSpans, FindValuePos

maxlen = 512

# FIXME: 
abl_norope = False # 默认为False

class VarLenLists:
    def __init__(self, prefix, dfile=None):
        self.prefix = prefix
        self.mem = []
        self.idx = [0]
        if dfile is not None:
            self.mem = dfile[f'{prefix}mem'][:]
            self.idx = dfile[f'{prefix}idx'][:]
    def append(self, xs):
        self.mem.extend(xs)
        self.idx.append(len(self.mem))
    def prepare(self):
        self.mem = np.array(self.mem)
        self.idx = np.array(self.idx)
    def save(self, dfile):
        self.prepare()
        prefix = self.prefix
        dfile.create_dataset(f'{prefix}mem', data=self.mem)
        dfile.create_dataset(f'{prefix}idx', data=self.idx)
    def __len__(self):
        return len(self.idx)-1
    def __getitem__(self, i):
        return self.mem[self.idx[i]:self.idx[i+1]]

class RelCombDataset(torch.utils.data.Dataset):
    def __init__(self, name, tokenizer, negative_threshold):
        self.tokenizer = tokenizer
        self.name = name
        datas = ljqpy.LoadJsons(name)
        self.items = []
        for textid, z in enumerate(tqdm(datas)):
            text = self.gettext(z)
            item = {'text': text}
            tids, otokens = self.tokenize(text)
            triels = set(); allels = set()
            for spo in self.getspos(z):
                s, o = self.gets(spo), self.geto(spo)
                if random.random() >= negative_threshold:
                    triels.add( (s, o) )
                allels.add(s)
                allels.add(o)
            uvs = {e:FindValuePos(otokens, e) for e in allels}
            item['id'] = textid
            item['otokens'] = otokens
            item['tids'] = torch.tensor(tids)
            item['uvs'] = uvs
            item['triels'] = triels
            self.items.append(item)

    def gettext(self, x): return x['sentText']
    def getspos(self, x): return x['relationMentions']
    def gets(self, x): return TN(x['em1Text'])
    def geto(self, x): return TN(x['em2Text'])
    def getr(self, x): return x['label']
    def tokenize(self, x): 
        tokens = self.tokenizer.tokenize(x)
        otokens = restore_token_list(x, tokens)
        tids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.sep_token_id]
        return tids, otokens

    def __len__(self): return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k]
        return item['tids'], item

def my_collate_fn_GP(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)[:,:maxlen]
    bsz, alen = xx.size()[:2]
    yy = torch.zeros((bsz, 1, alen, alen))
    for i, (x, item) in enumerate(items):
        uvs = item['uvs']
        for s, o in item['triels']:
            for u1, v1 in uvs[s]:
                for u2, v2 in uvs[o]:
                    for u in range(u1, v1):
                        for v in range(u2, v2):
                            if 1+u < alen and 1+v < alen: yy[i,0,1+u,1+v] = yy[i,0,1+v,1+u] = 1   
        for s, uv1 in uvs.items():
            for o, uv2 in uvs.items():
                if s == o or (s, o) in item['triels'] or (o, s) in item['triels']: continue
                for u1, v1 in uv1:
                    for u2, v2 in uv2:
                        for u in range(u1, v1):
                            for v in range(u2, v2):
                                if 1+u < alen and 1+v < alen: yy[i,0,1+u,1+v] = yy[i,0,1+v,1+u] = -1
    return xx, yy


class PositionalEncoding(nn.Module):
    # [bst, seq, fea]
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class GlobalPointerModel(nn.Module):
    def __init__(self, plm_name, heads=1, head_size=64):
        super().__init__()
        self.bert = BertModel.from_pretrained(plm_name)
        self.heads = heads
        self.head_size = head_size
        self.gpfc = nn.Linear(768, self.heads * self.head_size * 2)
        self.pe = PositionalEncoding(self.head_size)

    def get_gp_output(self, z):
        zsize = z.size()  # b x l x (heads * headsz * 2)
        gpin = z.view(zsize[0], zsize[1], self.heads, self.head_size, 2)
        qw, kw = gpin[...,0], gpin[...,1]

        if abl_norope:
            logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
            return logits / (self.head_size**0.5)

        pos = self.pe(qw)
        cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, -1)
        sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, -1)

        qw2 = torch.cat([-qw[..., 1::2, None], qw[..., ::2, None]], -1).view(qw.size())
        kw2 = torch.cat([-kw[..., 1::2, None], kw[..., ::2, None]], -1).view(kw.size())
        #kw2 = torch.cat([-qw[..., 1::2, None], qw[..., ::2, None]], -1).view(kw.size())
        qw = qw * cos_pos + qw2 * sin_pos
        kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        return logits / (self.head_size**0.5)
        
    def forward(self, x):
        z = self.bert(x).last_hidden_state
        return self.get_gp_output(self.gpfc(z))

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], -1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], -1)
    neg_loss = torch.logsumexp(y_pred_neg, -1)
    pos_loss = torch.logsumexp(y_pred_pos, -1)
    return neg_loss + pos_loss


def mcce3(y_pred, y_true):
    y_pred = - y_true * y_pred
    y_pred_neg = y_pred - (y_true > -0.5) * 1e12
    y_pred_pos = y_pred - (y_true < 0.5) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], -1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], -1)
    neg_loss = torch.logsumexp(y_pred_neg, -1)
    pos_loss = torch.logsumexp(y_pred_pos, -1)
    return neg_loss + pos_loss

def global_pointer_crossentropy(y_pred, y_true):
    bh = y_pred.size(0) * y_pred.size(1)
    y_pred = y_pred.view(bh, -1)
    y_true = y_true.view(bh, -1)
    return multilabel_categorical_crossentropy(y_pred, y_true).mean()

def gpce3(y_pred, y_true):
    bh = y_pred.size(0) * y_pred.size(1)
    y_pred = y_pred.view(bh, -1)
    y_true = y_true.view(bh, -1)
    return mcce3(y_pred, y_true).mean()

def bce3(y_pred, y_true):
    #z = - y_true*torch.log(y_pred+1e-9) - (1-y_true)*torch.log(1-y_pred+1e-9)  # BCE
    y_pred = torch.sigmoid(y_pred)
    z = - (y_true > 0.5).float() * torch.log(y_pred+1e-9)       # y_true == 1
    z += - (y_true < -0.5).float() * torch.log(1-y_pred+1e-9)   # y_true == -1
    return z.mean()

if __name__ == '__main__':    
    dss = {x:RelCombDataset(x) for x in ['test']}
    dls = {x:torch.utils.data.DataLoader(y, batch_size=3, shuffle=False, collate_fn=my_collate_fn_GP) for x,y in dss.items() if x != 'train'}

    model = GlobalPointerModel()
    for x, y in dls['test']:
        z = model(x)
        loss = gpce3(z, y)
        print(loss)
        break

    sys.exit()

'''
class CondExtDatasetGP(CondExtDataset):
    def __getitem__(self, k):
        item = self.items[k]
        textid, plen, ptid = self.prompts[k]
        x = torch.tensor(self.ptids[ptid].tolist()+self.tids[textid].tolist())
        y = [(plen+u,plen+v-1) for u, v in item['rpos']]
        return x, y

class CondExtDatasetPK(CondExtDataset):
    def __init__(self, name):
        self.name = name
        self.h5fn = f'cond/cache_pk_{name}.h5'
        self.load()

    def gen_prompts(self):
        global tokenizer
        if tokenizer is None: tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.prompts = np.zeros((len(self.items), 4), dtype=np.int32)
        self.ptids = VarLenLists('ptids')
        for ii, item in enumerate(tqdm(self.items)):
            prompt = f"{item['rkey']}|"
            conds = []
            for k, v in item['c'].items():
                conds.append(f'{k}:{v}')
            prompt += ','.join(conds)
            pt = tokenizer.encode(prompt)
            self.prompts[ii,0] = item['textid']
            self.prompts[ii,1] = len(pt)
            self.prompts[ii,2] = len(self.ptids)
            self.prompts[ii,3] = self.rkeyset[item['rkey']]
            self.ptids.append(pt)
        if not os.path.exists(self.h5fn): self.save()

    def __getitem__(self, k):
        item = self.items[k]
        textid, plen, ptid, rkey = self.prompts[k]
        x = torch.tensor(self.ptids[ptid].tolist()+self.tids[textid].tolist())
        y = [(plen+u,plen+v-1) for u, v in item['rpos']]
        return rkey, x, y


class CondMaskDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        self.name = name
        self.h5fn = f'cond/cache_mask_{name}.h5'
        self.mlen = 16
        self.prompts = None
        self.load()
            
    def load_data(self):
        name = self.name
        datas = ljqpy.LoadJsonsg(fns[name])
        self.texts = []
        self.otokens = []
        self.items = []
        self.tids = VarLenLists('tids')
        self.allrposs = VarLenLists('allrposs')
        for textid, z in enumerate(datas):
            self.texts.append(z['text'])
            self.otokens.append(z['otokens'])
            self.tids.append(z['tokenids'])
            allrposs = []
            for y in z['ys']:
                item = {'textid':textid, 'c':y['c'], 'rkey':y['rkey'], 
                        'rval':y['rval'], 'rpos':y['rpos']}
                for u, v in y['rpos']:
                    allrposs.extend([u,v])
                self.items.append(item)
            self.allrposs.append(allrposs)

        rkeyfile = 'cond/rkeys.txt'
        if os.path.exists(rkeyfile):
            self.rkeyset = ljqpy.LoadJsons(rkeyfile)[0]
        else:
            rkeyset = defaultdict(int)
            for item in self.items: rkeyset[item['rkey']] += 1
            self.rkeyset = {v[0]:k for k,v in enumerate(ljqpy.FreqDict2List(rkeyset))}
            ljqpy.SaveJsons([self.rkeyset], rkeyfile)

        self.tids.prepare()
        self.allrposs.prepare()
        if self.prompts is None: self.gen_prompts()

    def gen_prompts(self):
        global tokenizer
        if tokenizer is None: tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        mlen = self.mlen
        self.prompts = np.zeros((len(self.items), 2), dtype=np.int32)
        self.pconds = np.zeros((len(self.items), 5, mlen), dtype=np.int32)
        for ii, item in enumerate(tqdm(self.items)):
            for i, (k, v) in enumerate(item['c'].items()):
                pt = tokenizer.encode(f'{k}:{v}')[:mlen]
                self.pconds[ii,i,:len(pt)] = pt

            self.prompts[ii,0] = item['textid']
            self.prompts[ii,1] = self.rkeyset[item['rkey']]
        if not os.path.exists(self.h5fn): self.save()

    def save(self):
        retpos = VarLenLists('rpos')
        for item in self.items:
            xs = []
            for u, v in item['rpos']: xs.extend([u,v])
            retpos.append(xs)
        with h5py.File(self.h5fn, 'w') as dfile:
            self.tids.save(dfile)
            self.allrposs.save(dfile)
            retpos.save(dfile)
            dfile.create_dataset('prompts', data=self.prompts)
            dfile.create_dataset('pconds', data=self.pconds)

    def load(self):
        if os.path.exists(self.h5fn):
            with h5py.File(self.h5fn, 'r') as dfile:
                self.tids = VarLenLists('tids', dfile)
                self.allrposs = VarLenLists('allrposs', dfile)
                self.prompts = dfile['prompts'][:]
                self.pconds = dfile['pconds'][:]
                retpos = VarLenLists('rpos', dfile)
            self.items = [{'rpos':np.array(retpos[i]).reshape(-1,2)} for i in range(self.prompts.shape[0])]
        else:
            self.load_data()
                        
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        item = self.items[k]
        textid, rkey = self.prompts[k]
        cond = torch.tensor(self.pconds[k])
        x = torch.tensor([101]+self.tids[textid].tolist())
        y = [(1+u,v) for u, v in item['rpos']]
        ally = [(1+u,v) for u, v in self.allrposs[textid].reshape(-1, 2)]
        return (rkey, cond, x), (y, ally)

class AllYDataset(CondMaskDataset):
    def __len__(self): return len(self.tids)
    def __getitem__(self, k):
        x = torch.tensor([101]+self.tids[k].tolist())
        ally = [(1+u,v) for u, v in self.allrposs[k].reshape(-1, 2)]
        return x, ally

def my_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
    yy = nn.utils.rnn.pad_sequence([y for x,y in items], batch_first=True)
    return xx[:,:maxlen], yy[:,:maxlen].float()

def my_collate_fn_GP(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)[:,:maxlen]
    bsz, alen = xx.size()[:2]
    yy = torch.zeros((bsz, 1, alen, alen))
    for i, (x, y) in enumerate(items):
        for u, v in y: 
            if u < alen and v < alen: yy[i,0,u,v] = 1
    return xx, yy

def my_collate_fn_PK(items):
    rk = torch.tensor([x[0] for x in items]).unsqueeze(1)
    xx = nn.utils.rnn.pad_sequence([x[1] for x in items], batch_first=True)[:,:maxlen]
    bsz, alen = xx.size()[:2]
    yy = torch.zeros((bsz, 1, alen, alen))
    for i, x in enumerate(items):
        for u, v in x[2]: 
            if u < alen and v < alen: yy[i,0,u,v] = 1
    return rk, xx, yy

def my_collate_fn_CM(items):
    rk = torch.tensor([x[0][0] for x in items]).unsqueeze(1)
    cond = torch.cat([x[0][1].unsqueeze(0) for x in items], dim=0)
    xx = nn.utils.rnn.pad_sequence([x[0][2] for x in items], batch_first=True)[:,:maxlen]
    bsz, alen = xx.size()[:2]
    yy = torch.zeros((bsz, 2, alen, alen))
    for i, x in enumerate(items):
        for u, v in x[1][0]: 
            if u < alen and v < alen: yy[i,0,u,v] = 1
        for u, v in x[1][1]: 
            if u < alen and v < alen: yy[i,1,u,v] = 1
    return (rk, cond, xx), yy


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.fc = nn.Linear(768, 2)
        self.softmax = False

    def forward(self, x):
        z = self.bert(x).last_hidden_state
        out = self.fc(z)
        if self.softmax:
            out = F.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out)
        return out

def DecodeBase(ds, kk, out):
    textid, plen, _ = ds.prompts[kk]
    out = out[plen:]
    topspans = GetTopSpans(ds.otokens[textid], out)
    extracted = [x for x,y in topspans if y > threshold]
    return extracted

def DecodeGP(ds, kk, out):
    textid, plen = ds.prompts[kk][:2]
    extracted = []
    alen = out.shape[-1]
    out = out[0]
    linemax = out.max(-1)
    for i in range(plen, alen):
        if linemax[i] < 0: continue
        for j in range(i, alen):
            if out[i,j] > 0: extracted.append(''.join(ds.otokens[textid][i-plen:j-plen+1]).strip())
    return list(set(extracted))

def DecodeAllY(ds, kk, out):
    textid, plen = kk, 1
    extracted = []
    alen = out.shape[-1]
    out = out[0]
    linemax = out.max(-1)
    for i in range(plen, alen):
        if linemax[i] < 0: continue
        for j in range(i, alen):
            if out[i,j] > 0: extracted.append(''.join(ds.otokens[textid][i-plen:j-plen+1]).strip())
    return list(set(extracted))

def DecodeCM(ds, kk, out):
    textid, plen = ds.prompts[kk][0], 1
    extracted = []
    alen = out.shape[-1]
    out = out[0]
    linemax = out.max(-1)
    for i in range(plen, alen):
        if linemax[i] < 0: continue
        for j in range(i, alen):
            if out[i,j] > 0: 
                extracted.append(''.join(ds.otokens[textid][i-plen:j-plen+1]).strip())
    return list(set(extracted))

def sequence_masking(x, mask, value=0.0, axis=1):
    if mask is None: return x
    for _ in range(axis-1): 
        mask = torch.unsqueeze(mask, 1)
    for _ in range(x.dim() - mask.dim()): 
        mask = torch.unsqueeze(mask, mask.dim())
    return x * mask + value * (1 - mask)


class GlobalPointerModel(nn.Module):
    def __init__(self, heads=1, head_size=64):
        super().__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.heads = heads
        self.head_size = head_size
        self.gpfc = nn.Linear(768, self.heads * self.head_size * 2)
        self.pe = PositionalEncoding(self.head_size)

    def get_gp_output(self, z, mask):
        zsize = z.size()  # b x l x (heads * headsz * 2)
        gpin = z.view(zsize[0], zsize[1], self.heads, self.head_size, 2)
        qw, kw = gpin[...,0], gpin[...,1]

        pos = self.pe(qw)
        cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, -1)
        sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, -1)

        qw2 = torch.cat([-qw[..., 1::2, None], qw[..., ::2, None]], -1).view(qw.size())
        kw2 = torch.cat([-qw[..., 1::2, None], qw[..., ::2, None]], -1).view(kw.size())
        qw = qw * cos_pos + qw2 * sin_pos
        kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        if mask:
            logits = sequence_masking(logits, mask, -1e9, 2)
            logits = sequence_masking(logits, mask, -1e9, 3)

        trimask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - trimask * 1e11
        return logits / (self.head_size**0.5)
        
    def forward(self, x, mask=None):
        z = self.bert(x).last_hidden_state
        return self.get_gp_output(self.gpfc(z), mask)


class PromptKeyModel(GlobalPointerModel):
    def __init__(self, heads=1, head_size=64):
        super().__init__(heads=heads, head_size=head_size)
        self.rkeyemb = nn.Embedding(10, 768)
        
    def encode(self, rkey, x):
        embedding_output = self.bert.embeddings(input_ids=x)
        rkeyc = self.rkeyemb(rkey)
        inneremb = torch.cat([rkeyc, embedding_output], dim=1)
        encoder_outputs = self.bert.encoder(inneremb)
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output[:,1:]
        return sequence_output

    def forward(self, rkey, x):
        z = self.encode(rkey, x)
        return self.get_gp_output(self.gpfc(z), mask=None)

class CondMaskModel(GlobalPointerModel):
    def __init__(self, heads=1, head_size=64):
        super().__init__(heads=heads, head_size=head_size)
        self.rkeyemb = nn.Embedding(10, 768)

        encoder_layer = nn.TransformerEncoderLayer(768, nhead=8, dim_feedforward=768*2, dropout=0.1, batch_first=True)
        encoder_norm = nn.LayerNorm(768, eps=1e-5)

        self.encoder = nn.TransformerEncoder(encoder_layer, 2, encoder_norm)

    def forward(self, rkey, cond, x):
        sent_emb = self.bert.embeddings(input_ids=x)
        sent_out = self.bert.encoder(sent_emb)[0]   # [bsz, len, 768]

        yout = self.gpfc(sent_out)
        yall = self.get_gp_output(yout, mask=None)

        rkeyc = self.rkeyemb(rkey)  # [bsz, 1, 768]
        rkinp = torch.cat([rkeyc, sent_out], dim=1)
        rkmasks = self.encoder(rkinp)[:,1:]

        bsz, cnum, clen = cond.size()[:3]
        condflat = cond.view(bsz*cnum, clen)  # [bsz*5, 36]
        cemb = self.bert.embeddings(input_ids=condflat)
        semb = torch.repeat_interleave(sent_out.unsqueeze(1), cnum, dim=1)
        sembflat = semb.view(bsz*cnum, -1, 768)

        cinp = torch.cat([cemb, sembflat], dim=1)

        cmasks = self.encoder(cinp)[:,clen:]
        cmasks = cmasks.view(bsz, cnum, -1, 768)

        allmasks = torch.cat([rkmasks.unsqueeze(1), cmasks], dim=1)
        allmasks = torch.min(allmasks, dim=1, keepdim=True)[0]

        condout = torch.cat([sent_out.unsqueeze(1), allmasks], dim=1)
        condout = torch.min(allmasks, dim=1)[0]

        yfin = self.get_gp_output(self.gpfc(condout), mask=None)
        return yfin, yall
        

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], -1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], -1)
    neg_loss = torch.logsumexp(y_pred_neg, -1)
    pos_loss = torch.logsumexp(y_pred_pos, -1)
    return neg_loss + pos_loss

def global_pointer_crossentropy(y_pred, y_true):
    bh = y_pred.size(0) * y_pred.size(1)
    y_pred = y_pred.view(bh, -1)
    y_true = y_true.view(bh, -1)
    return multilabel_categorical_crossentropy(y_pred, y_true).mean()

def global_pointer_f1_score(y_pred, y_true):
    y_pred = (y_pred > 0).float()
    return 2 * (y_true * y_pred).sum() / (y_true + y_pred).sum()

def GetTopSpans(tokens, rr, K=40):
    cands = defaultdict(float)
    start_indexes = sorted(enumerate(rr[:,0]), key=lambda x:-x[1])[:K]
    end_indexes = sorted(enumerate(rr[:,1]), key=lambda x:-x[1])[:K]
    for start_index, start_score in start_indexes:
        if start_score < 0.1: continue
        if start_index >= len(tokens): continue
        for end_index, end_score in end_indexes:
            if end_score < 0.1: continue
            if end_index >= len(tokens): continue
            if end_index < start_index: continue
            length = end_index - start_index + 1
            if length > 40: continue
            ans = ''.join(tokens[start_index:end_index+1]).strip()
            if '》' in ans: continue
            if '、' in ans and len(ans.split('、')) > 2 and '，' not in ans and ',' not in ans:
                aas = ans.split('、')
                for aa in aas: cands[aa.strip()] += start_score * end_score / len(aas)
                continue
            cands[ans] += start_score * end_score
    cand_list = sorted(cands.items(), key=lambda x:len(x[0]))
    removes = set()
    contains = {}
    for i, (x, y) in enumerate(cand_list):
        for j, (xx, yy) in enumerate(cand_list[:i]):
            if xx in x and len(xx) < len(x):
                contains.setdefault(x, []).append(xx)

    for i, (x, y) in enumerate(cand_list):
        sump = sum(cands[z] for z in contains.get(x, []) if z not in removes)
        suml = sum(len(z) for z in contains.get(x, []) if z not in removes)
        if suml > 0: sump = sump * min(1, len(x) / suml)
        if sump > y: removes.add(x)
        else:
            for z in contains.get(x, []): removes.add(z)

    ret = [x for x in cand_list if x[0] not in removes]
    ret.sort(key=lambda x:-x[1])
    return ret[:K]

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

if __name__ == '__main__':
    if 'base' in sys.argv:
        TheDataset = CondExtDataset
        Decode = DecodeBase
    elif 'gp' in sys.argv:
        TheDataset = CondExtDatasetGP
        my_collate_fn = my_collate_fn_GP
        Decode = DecodeGP
    elif 'pk' in sys.argv:
        TheDataset = CondExtDatasetPK
        my_collate_fn = my_collate_fn_PK
        Decode = DecodeGP
    elif 'cm' in sys.argv:
        TheDataset = CondMaskDataset
        my_collate_fn = my_collate_fn_CM
        Decode = DecodeCM
    elif 'ally' in sys.argv:
        TheDataset = AllYDataset
        my_collate_fn = my_collate_fn_GP
        Decode = DecodeAllY

    tic = time.time()
    dss = {x:TheDataset(x) for x in 'valid test'.split()}
    dls = {x:torch.utils.data.DataLoader(y, batch_size=20, shuffle=False, collate_fn=my_collate_fn) for x,y in dss.items() if x != 'train'}
    print(f'data loaded, {time.time()-tic:.3f} sec.')
    
    testtp = 'test' if 'valid' not in sys.argv else 'valid'
    ds = dss[testtp]
    ds.load_data()

    if 'base' in sys.argv:
        model = BaseModel().cuda()
        mfile = 'cond/basemodel.pth'
        model.load_state_dict(torch.load(mfile))
    elif 'gp' in sys.argv:
        model = GlobalPointerModel().cuda()
        mfile = 'cond/gpmodel.pth'
        model.load_state_dict(torch.load(mfile))
    elif 'pk' in sys.argv:
        model = PromptKeyModel().cuda()
        mfile = 'cond/pkmodel.pth'
        model.load_state_dict(torch.load(mfile))
    elif 'cm' in sys.argv:
        model = CondMaskModel().cuda()
        mfile = 'cond/cmmodel.pth'
        model.load_state_dict(torch.load(mfile))
    elif 'ally' in sys.argv:
        model = GlobalPointerModel().cuda()
        mfile = 'cond/allymodel.pth'
        with torch.no_grad():
            k = 0
            for xx, yy in dls['test']:
                print(xx.size(), yy.size())
                zz = model(xx.cuda())
                loss = global_pointer_crossentropy(zz.cpu(), yy)
                print(loss)
                print(Decode(dss['test'], k, yy[0].numpy()))
                k += 20
        sys.exit()
        
    print('model loaded!')
    allouts = []
    with torch.no_grad():
        if 'pk' in sys.argv:
            for rk, xx, yy in tqdm(dls[testtp]):
                out = model(rk.cuda(), xx.cuda()).cpu().numpy()
                for x in out: allouts.append(x)
        elif 'cm' in sys.argv:
            for (rk, cond, xx), yy in tqdm(dls[testtp]):
                out = model(rk.cuda(), cond.cuda(), xx.cuda())[0].cpu().numpy()
                for x in out: allouts.append(x)
        else:
            for xx, yy in tqdm(dls[testtp]):
                out = model(xx.cuda()).cpu().numpy()
                for x in out: allouts.append(x)
    f1, spf1 = MetricF1(), MetricF1()
    threshold = 0.5
    with open('record.txt', 'w', encoding='utf-8') as fout:
        for kk, out in enumerate(allouts):
            textid, plen = ds.prompts[kk][:2]
            golden = ds.items[kk]['rval']
            extracted = Decode(ds, kk, out)
            f1.append(extracted, golden)
            if ds.items[kk]['rkey'] not in 'so': spf1.append(extracted, golden)
            ljqpy.WriteLine(fout, [extracted, golden, ds.items[kk]['rkey'], ds.items[kk]['c'], ds.texts[textid]])
    f1.compute()
    print('special fields:')
    spf1.compute()


'''