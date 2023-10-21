#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import *
#from tensorflow.keras.callbacks import *
#from bert4keras.backend import keras, K
from collections import defaultdict
import numpy as np
import re, unicodedata

def dgcnn_block(x, dim, dila=1):
    y1 = Conv1D(dim, 3, padding='same', dilation_rate=dila)(x)
    y2 = Conv1D(dim, 3, padding='same', dilation_rate=dila, activation='sigmoid')(x)
    yy = multiply([y1, y2])
    if yy.shape[-1] == x.shape[-1]: yy = add([yy, x])
    return yy
    
def neg_log_mean_loss(y_true, y_pred):
    eps = 1e-6
    pos = - K.sum(y_true * K.log(y_pred+eps), 1) / K.maximum(eps, K.sum(y_true, 1))
    neg = K.sum((1-y_true) * y_pred, 1) / K.maximum(eps, K.sum(1-y_true, 1))
    neg = - K.log(1 - neg + eps)
    return K.mean(pos + neg * 10)


def TN(token):
    return re.sub('[\u0300-\u036F]', '', unicodedata.normalize('NFKD', token))

def gen_token_list_inv_pointer(sent, token_list):
	sent = sent.lower()
	otiis = []; iis = 0 
	for it, token in enumerate(token_list):
		otoken = token.lstrip('#').lower()
		if token[0] == '[' and token[-1] == ']': otoken = ''
		niis = iis
		while niis <= len(sent):
			if sent[niis:].startswith(otoken): break
			if niis >= len(sent): break
			if otoken in '-"' and sent[niis][0] in '—“”': break
			niis += 1
		if niis >= len(sent): niis = iis
		otiis.append(niis)
		iis = niis + max(1, len(otoken))
	return otiis

# restore [UNK] tokens to the original tokens
def restore_token_list(sent, token_list):
	if token_list[0] == '[CLS]': token_list = token_list[1:-1]
	invp = gen_token_list_inv_pointer(sent, token_list)
	invp.append(len(sent))
	otokens = [sent[u:v] for u,v in zip(invp, invp[1:])]
	processed = -1
	for ii, tk in enumerate(token_list):
		if tk != '[UNK]': continue
		if ii < processed: continue
		for jj in range(ii+1, len(token_list)):
			if token_list[jj] != '[UNK]': break
		else: jj = len(token_list)
		allseg = sent[invp[ii]:invp[jj]]

		if ii + 1 == jj: continue
		seppts = [0] + [i for i, x in enumerate(allseg) if i > 0 and i+1 < len(allseg) and x == ' ' and allseg[i-1] != ' ']
		if allseg[seppts[-1]:].replace(' ', '') == '': seppts = seppts[:-1]
		seppts.append(len(allseg))
		if len(seppts) == jj - ii + 1:
			for k, (u,v) in enumerate(zip(seppts, seppts[1:])): 
				otokens[ii+k] = allseg[u:v]
		processed = jj + 1
	if invp[0] > 0: otokens[0] = sent[:invp[0]] + otokens[0]
	if ''.join(otokens) != sent:
		raise Exception('restore tokens failed, text and restored:\n%s\n%s' % (sent, ''.join(otokens)))
	return otokens

def FindValuePos(sent, value):
    ret = [];  
    value = value.replace(' ', '').lower()
    if value == '': return ret
    ss = [x.replace(' ', '').lower() for x in sent]
    for k, v in enumerate(ss):
        if not value.startswith(v): continue
        vi = 0
        for j in range(k, len(ss)):
            if value[vi:].startswith(ss[j]):
                vi += len(ss[j])
                if vi == len(value):
                    ret.append( (k, j+1) )
            else: break
    return ret

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

def batch_generator(gen, batch_size):
    X = next(gen)
    batch = [np.zeros((batch_size,)+x.shape[1:], dtype=x.dtype) for x in X]
    ii = 0
    while True:
        X = next(gen)
        N = X[0].shape[0]
        for i in range(N):
            for k, x in enumerate(X):
                batch[k][ii] = x[i]
            ii += 1
            if ii >= batch_size:
                yield [x.copy() for x in batch[:-1]], batch[-1].copy()
                ii = 0
     