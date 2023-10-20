import os, sys, time, ljqpy
from random import sample, shuffle
import torch
import torch.nn as nn
import transformers
from transformers import set_seed
from tqdm import tqdm
from transformers import BertTokenizer
import argparse
from accelerate import Accelerator, DistributedDataParallelKwargs
from relcomb import *
from config import config

accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
parser = argparse.ArgumentParser()
parser.add_argument("--dname", help='dataset for training and testing', choices=['ske2019','HacRED','NYT10-HRL','NYT11-HRL','NYT21-HRL', 'WebNLG', 'NYT10', 'WikiKBP', 'WebNLG_star','CoNLL04'], default='HacRED')
parser.add_argument("--negative_rate", help="negative ratio", default=0, type=float)
args = parser.parse_args()
negative_threshold = args.negative_rate

dname = args.dname
# set_seed(52)
# set_seed(58)  # new1
set_seed(77)  # new2

datadir = './dataset/' + dname
dsplits = 'train test valid'.split()
fns = {x:os.path.join(datadir, f'new_{x}.json') for x in dsplits}

# plm_name = config[dname]['plm_name']
if dname in ['HacRED', 'ske2019']: plm_name = '/data/dell/dingzepeng/hub/models--hfl--chinese-roberta-wwm-ext/snapshots/5c58d0b8ec1d9014354d691c538661bf00bfdb44'
# if dname in ['HacRED', 'ske2019']: plm_name = 'bert-base-chinese'
else: plm_name = '/data/dell/dingzepeng/hub/models--bert-base-cased/snapshots/5532cc56f74641d4bb33641f5c76a55d11f846e0'
tokenizer = BertTokenizer.from_pretrained(plm_name)

if not os.path.isdir(dname): os.makedirs(dname)
def wdir(x): return os.path.join(dname, x)

loadold = False

tic = time.time()
dss = {x:RelCombDataset(fns[x], tokenizer, negative_threshold if x == 'train' else 0) for x in ['train','valid']}
dls = {'train':torch.utils.data.DataLoader(dss['train'], batch_size=16, shuffle=True, collate_fn=my_collate_fn_GP),
	   'valid':torch.utils.data.DataLoader(dss['valid'], batch_size=16, shuffle=True, collate_fn=my_collate_fn_GP)}
print(f'data loaded, {time.time()-tic:.3f} sec.')

model = GlobalPointerModel(plm_name)
mfile = wdir(f'relcomb_BCE_sh_{negative_threshold}.pt')
# mfile = wdir(f'relcomb_BCE_abl_norope.pt')

sys.path.append('../')
import pt_utils

epochs = 20
total_steps = len(dls['train']) * epochs

optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 1e-5, total_steps)
model, optimizer, dls['train'], dls['valid'] = accelerator.prepare(model, optimizer, dls['train'], dls['valid'])
def train_func(model, ditem):
	xx, yy = ditem[0], ditem[1]
	zz = model(xx)
	#loss = gpce3(zz, yy)
	loss = bce3(zz, yy)
	return {'loss': loss}

""" def test_func(model, ditem): 
	with torch.no_grad():
		xx, yy = ditem[0], ditem[1]
		zz = model(xx)
		loss = bce3(zz, yy)
	return {'loss': loss} """
def test_func(): pass

if __name__ == '__main__':
	if loadold: model.load_state_dict(torch.load(mfile))
	pt_utils.train_model(model, optimizer, dls['train'],epochs, train_func, test_func, 
				scheduler=scheduler, save_file=mfile, accelerator=accelerator)
