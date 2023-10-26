import os, sys, time, ljqpy, math, re, json
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dname", help='dataset for testing', choices=['ske2019','HacRED','NYT10-HRL','NYT11-HRL','NYT21-HRL', 'WebNLG', 'WikiKBP', 'NYT10', 'WebNLG_star','CoNLL04'], default='HacRED')
parser.add_argument("--model", help='LLM model name', choices=['llama','qwen7B','vicuna'], default='qwen7B')
parser.add_argument("--peft", help='whether peft or not', type=bool, default=True)
parser.add_argument("--stage", help='which stage to delete', choices=['one','two'], default='one')

args = parser.parse_args()
dname = args.dname
datadir = './dataset/' + dname
def wdir(x): return os.path.join(dname, x)
if args.stage == 'two':
    if args.peft:
        origin_file = 'your path'
    else:
        origin_file = 'your path'
else:
    if args.peft:
        origin_file = 'your path'
    else:
        origin_file = 'your path'

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
    return ans

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
    
with open(wdir(origin_file),'r',encoding='utf-8') as fin:
    preds = json.load(fin)

if args.stage == 'two':
    with open(wdir('your_path'),'r',encoding='utf-8') as fin:
        outs = json.load(fin)
    for i in range(min(len(outs),len(preds))):
        add_list = outs[i]['preds']
        preds[i]['relationMentions'] = preds[i]['relationMentions'] + add_list
else:
    with open(wdir('your_path'),'r',encoding='utf-8') as fin:
        outs = json.load(fin)
    for i in range(min(len(outs),len(preds))):
        candi_list = outs[i]['preds']
        filtered = []
        for itm in preds[i]['relationMentions']:
            dic1 = {'em1Text':itm['em1Text'], 'em2Text':itm['em2Text'] if 'em2Text' in itm.keys() else ''}
            if dic1 in candi_list:
                filtered.append(itm)
        preds[i]['relationMentions'] = filtered

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

spo_limits = [1,2]
for spo_limit in spo_limits:
    f1 = MetricF1()
    fout = open('your_path', 'w', encoding='utf-8')
    for i in range(min(len(outs),len(preds))):
        if len(outs[i]['std_ans']) >= spo_limit:
            ComputeOne_2(outs[i], preds[i], f1, fout)
    print('spo_limit:{}\n'.format(spo_limit))
    f1.compute()
    fout.close()