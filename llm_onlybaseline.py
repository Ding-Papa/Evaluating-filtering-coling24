import os, sys, time, ljqpy, math, re, json
import unicodedata
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
parser.add_argument("--dname", help='dataset for training and testing', choices=['ske2019','HacRED','NYT10-HRL','NYT11-HRL','NYT21-HRL', 'WebNLG', 'WikiKBP', 'NYT10', 'WebNLG_star','CoNLL04'], default='HacRED')
args = parser.parse_args()
dname = args.dname
datadir = './dataset/' + dname
dsplits = 'train test valid'.split()
fns = {x:os.path.join(datadir, f'new_{x}.json') for x in dsplits}

def wdir(x): return os.path.join(dname, x)
with open(os.path.join(datadir, 'rel2id.json')) as fin:
    rel_map = json.load(fin)
rev_rel_map = {v:k for k,v in rel_map.items()}
relation_list = list(rel_map.keys())

inst = f'预定义好下列关系列表：{relation_list}，请从下面的句子中抽取出包含上述关系的所有三元组。'+\
    '''
    注意，三元组的关系名称必须从上面的关系列表中选取，不考虑除此以外的其他关系。请按照下面指定的格式进行输出：
    {"relationMentions": [{"em1Text": 主体1, "em2Text": 客体1, "label": 关系1}, {"em1Text": 主体2, "em2Text": 客体2, "label": 关系2}]}
    注意，三元组并不一定只有两个，请你模仿这个格式，将所有符合要求的三元组都输出出来。
    下面是一个例子：
    输入：周佛海被捕入狱之后，其妻杨淑慧散尽家产请蒋介石枪下留人，于是周佛海从死刑变为无期，不过此人或许作恶多端，改判没多久便病逝于监狱，据悉是心脏病发作
    输出：{ "relationMentions": [{"em1Text": "周佛海", "em2Text": "杨淑慧", "label": "妻子"}, {"em1Text": "杨淑慧", "em2Text": "周佛海", "label": "丈夫"}]}

    再次强调，所输出的三元组的关系必须从上面所给的预定义列表中选取，不得输出任何不在列表中的关系。同时，请尽可能多地输出符合要求的三元组。
    接下来请模仿这个例子，根据输入，按格式要求输出包含上述关系的所有三元组。注意当实体（主体或客体）可以拆分为两个词语（比如中间有顿号或逗号）时，其应当被拆分为两个三元组而不是合并在一个三元组内。\n->
    '''
inst_english = f'Predefine the following relationship list:{relation_list}, please extract all triples containing the above relationship from the following sentences.'+\
    '''
    Note that the relationship name of the triple must be selected from the above relationship list, and other relationships not listed are not considered. Please output according to the specified format below:
    [{"em1Text": subject1, "em2Text": object1, "label": relationship1}, {"em1Text": subject2, "em2Text": object2, "label": relationship2}]
    Note that the triple may not only have two, please imitate this format and output all triples that meet the requirements.
    Here is an example:
    Input: For much of the three years since the show first opened at Yale , Peru 's claim on the objects has been played out in behind-the-scenes talks in Lima , Washington and New Haven between Yale and the government of President Alejandro Toledo .
    Output: [{"em1Text": "Alejandro Toledo", "em2Text": "Peru", "label": "/people/person/nationality"}, {"em1Text": "Peru", "em2Text": "Lima", "label": "/location/location/contains"}, {"em1Text": "Peru", "em2Text": "Lima", "label": "/location/country/capital"}]

    Again, it is emphasized that the relationship of the triples output must be selected from the predefined list above, and no relationship not in the list can be output. At the same time, please output as many triples as possible that meet the requirements.
    Please imitate this example, according to the input, output all triples containing the above relationship according to the format requirements. Note that when the entity (subject or object) can be split into two words (such as a comma or comma in the middle), it should be split into two triples instead of merging into one triple.\n->
    '''

# inst_english = f'Predefine the following relationship list:{relation_list}, please extract all triples containing the above relationship from the following sentences.'+\
#     '''
#     Note that the relationship name of the triple must be selected from the above relationship list, and other relationships not listed are not considered. Please output according to the specified format below:
#     [{"em1Text": subject1, "em2Text": object1, "label": relationship1}, {"em1Text": subject2, "em2Text": object2, "label": relationship2}]
#     Note that the triple may not only have two, please imitate this format and output all triples that meet the requirements.
#     Again, it is emphasized that the relationship of the triples output must be selected from the predefined list above, and no relationship not in the list can be output. At the same time, please output as many triples as possible that meet the requirements.
#     Please according to the input, output all triples containing the above relationship according to the format requirements. Note that when the entity (subject or object) can be split into two words (such as a comma or comma in the middle), it should be split into two triples instead of merging into one triple.\n->
#     '''

if __name__ == "__main__":
    def generate_prompt(query, history, input=None):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "{}{}\n<end>".format(old_query, response)
        prompt += "{}".format(query)
        return prompt
    model_name = 'your_base_model_path'
    path_to_adapter = 'your_adapter_path(Qwen)'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(path_to_adapter,trust_remote_code=True).cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).cuda()
    # model = PeftModel.from_pretrained(model, 'your_peft_model_path', fan_in_fan_out=False)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.eval()
    model = model.to(device)
    
    def general_llm(wdata):
        query_session = inst_english + '\n' + wdata['sentText'] + '\n->'
        history = []
        prompt = generate_prompt(query_session, history)
        input_ids = tokenizer(prompt, return_tensors="pt")
        input_ids = input_ids["input_ids"].to(device)
        with torch.no_grad():
            outputs=model.generate(
                    input_ids=input_ids,
                    top_p=0.8,
                    top_k=50,
                    repetition_penalty=1.1,
                    max_new_tokens = 512,
                    early_stopping = True,
                    eos_token_id = tokenizer.convert_tokens_to_ids('<end>'),
                    pad_token_id = tokenizer.eos_token_id,
                    min_length = input_ids.shape[1] + 1
            )
        s = outputs[0]
        # response = tokenizer.decode(s[len(input_ids[0]):])
        response = tokenizer.decode(s)
        response = response.replace('<s>', '').replace('<end>', '').replace('</s>', '')
        return response
    
    origin_list = []
    idx = 0
    
    with open(wdir('your_file_path_name'), 'r', encoding='utf-8') as fin:
        wdatas = json.load(fin)
        for wdata in tqdm(wdatas):
            try:
                time.sleep(3)
                p1 = general_llm(wdata)
                origin_list.append(p1)
            except:
                time.sleep(10)
                p1 = general_llm(wdata)
                origin_list.append(p1)
            idx += 1
            if idx % 5 == 0:
                with open(wdir('your_file_path_name'), 'a', encoding='utf-8') as fout:
                    json.dump(origin_list, fout, ensure_ascii=False, indent=2)
                origin_list = []  
