import os, sys, json, random, pickle
import torch
from torch.utils.data import Dataset

meta_prompt = ''

def MakeSFTTokens(item, tokenizer):
    Q, A = item.get('instruction', '')+item.get('input', ''), item['output']
    S1 = Q
    ids = tokenizer(S1)['input_ids']
    labels = [-100] * len(ids)
    S2 = A + tokenizer.eos_token
    t2 = tokenizer(S2, add_special_tokens=False)['input_ids']
    ids += t2; labels += t2
    return ids, labels

def MixIFTGenerator(qas, odata, tokenizer, ratio=1):
    M = len(qas)
    N = int(len(qas) * (1 + ratio))
    qaids = [i for i in range(M)] + [-1] * (N - M)
    while True:
        random.shuffle(qaids)
        for ii in qaids:
            if ii < 0: item = random.choice(odata)
            else: item = qas[ii]
            ids, labels = MakeSFTTokens(item, tokenizer)
            if max(labels) < 0: continue
            yield ids, labels

def IFTBuffer(gen, length=512):
    buffer = []; buffer2 = []
    while True:
        ids, labels = next(gen)
        while len(ids) > 0:
            remaining = length - len(buffer)
            if remaining > 0:
                buffer.extend(ids[:remaining])
                buffer2.extend(labels[:remaining])
                ids = ids[remaining:]
                labels = labels[remaining:]
            if len(buffer) >= length: 
                if max(buffer2) > 0:
                    yield torch.tensor(buffer), torch.tensor(buffer2)
                buffer = []; buffer2 = []            

class PureGenDataset(Dataset):
    def __init__(self, gen, num=10000000) -> None:
        self.gen = gen
        self.num = num
    def __len__(self): return self.num
    def __getitem__(self, k):
        return next(self.gen)