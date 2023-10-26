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