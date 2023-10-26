import os, sys, argparse, json, time, random
import torch
import deepspeed
import torch.distributed as dist
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import pickle
from finetune_func import train, prepare, get_model_layers

from peft import (
    get_peft_model,
    PeftModel
)
from config import lora_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument("--max_epoches",type=int,default=30,help="max epoches to run dataloader")
    parser.add_argument("--max_training_samples",type=int,default=-1,help="max number of training samples")
    parser.add_argument("--epoch_steps",type=int,default=300,help="the number of steps for each epoch")
    parser.add_argument("--model_path",type=str,default='',help="the floader to load model")
    parser.add_argument("--max_length",type=int,default=1024,help="max token length")
    parser.add_argument("--use_flash_attention",action="store_true",help="whether to use flash attention")
    parser.add_argument("--use_lora",action="store_true",help="Whether to use LoRa, the default is to perform full Finetune")
    parser.add_argument("--load_lora",action="store_true",help="whether load ckpts")
    parser.add_argument("--load_lora_path",type=str,default="",help="the folder to load lora ckpts(.pt)")
    parser.add_argument("--save_dir",type=str,default="vicuna/",help="the folder to save ckpts(.pt)")
    parser.add_argument("--save_name",type=str,default="NYT10",help="the floader extension name")
    parser.add_argument("--save_steps",type=int,default=1000,help="how many step to save a model")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    prepare(args)

    args.model_path = '/data/Logic/vicuna-13b-v1.5'
    model_name = args.model_path
    print('model_name:', model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if not args.use_lora:
        st = "<end>"
        if st not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + st})
            print('additional_special_tokens:', tokenizer.additional_special_tokens)
    
    model = LlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True)

    if not args.use_lora:
        if model.config.vocab_size != len(tokenizer):
            print('resize token embeddings...')
            model.resize_token_embeddings(len(tokenizer))
            model_layers = get_model_layers(model)
            for layer in model_layers:
                if layer[0] in ['model.embed_tokens']:
                    begin_idx = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens[0])
                    end_idx = begin_idx + len(tokenizer.additional_special_tokens)
                    print('normalize special token...')
                    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.tensor([begin_idx]))))
                    torch.nn.init.normal_(layer[1].weight.data[begin_idx:end_idx], std=1e-6)
    else:
        if args.load_lora:
            print('lora parameter loaded!')
            print(args.load_lora_path)
            model = PeftModel.from_pretrained(model, args.load_lora_path, is_trainable= True)
        else:
            print('training from scratch')
            model = get_peft_model(model, lora_config)

    sys.path.append('../')

    from sft_dataset import PureGenDataset, MixIFTGenerator, IFTBuffer
    data = pickle.load(open('/data/dell/ljq/inst_dataset/623v1/ift_data.pkl', 'rb'))  # 通用数据，后面要和mydata mix以免大模型丢掉通用泛化能力
    data = [x[0] for x in data]    # 1164743条

    dname = args.save_name
    datadir = '../dataset/' + dname
    dsplits = 'train test valid'.split()
    with open(os.path.join(datadir, 'rel2id.json')) as fin:
        rel_map = json.load(fin)
    relation_list = list(rel_map.keys())
    
    # Next step: 融合构造训练数据，包含无candidate pair和没有candidate pair的两种数据，一起fine-tune
    inst_english = f'Predefine the following relationship list:{relation_list}, please extract all triples containing the above relationship from the following sentences.'+\
    '''
    Note that the relationship name of the triple must be selected from the above relationship list, and other relationships not listed are not considered. Please output according to the specified format below:
    [{"em1Text": subject1, "em2Text": object1, "label": relationship1}, {"em1Text": subject2, "em2Text": object2, "label": relationship2}]
    Note that the triple may not only have two, please imitate this format and output all triples that meet the requirements.
    Again, it is emphasized that the relationship of the triples output must be selected from the predefined list above, and no relationship not in the list can be output. At the same time, please output as many triples as possible that meet the requirements.
    Please according to the input, output all triples containing the above relationship according to the format requirements. Note that when the entity (subject or object) can be split into two words (such as a comma or comma in the middle), it should be split into two triples instead of merging into one triple.\n->
    '''
    # mydata = ljqpy.LoadJsons(os.path.join(datadir, 'new_train.json'))
    # mydata = [{'instruction':inst_english + '\n', 'input':x['sentText'], 'output':str(x['relationMentions'])} for x in mydata[:200]]
    with open('../'+dname+'/candidate.json','r',encoding='utf-8') as fin:
        candi = json.load(fin)
    idx=0
    mydata=[]
    for x in candi[3900:]:  # 前800个用来微调
        idx+=1
        if idx%1==0:  # 无filter:有filter=1:1
            mydata.append({'instruction':inst_english + '\n', 'input':x['sentText'] + '\n->', 'output':str(x['std_ans'])})
        else:
            candidates = [(c['em1Text'], c['em2Text']) for c in x['preds']]
            # inst_filter = f'经过检测，上面句子中可能有关系的实体对为{candidates}。请对抽取结果进行检查，并把漏掉的三元组补齐，删除错误的三元组，输出最终结果。'
            inst_filter = f'Now we claim that the entity pairs that may be related in the above sentence are {candidates}. Please check the extraction results and fill in the missing triples, remove the wrong triples and output the final result.\n->'
            pseudo_ext = [trip for trip in x['std_ans'] if x['std_ans'].index(trip)%2==0]   # 模拟大模型没有给提示的时候只抽出来一半
            mydata.append({'instruction':inst_english + '\n', 'input':x['sentText'] + '\n->' + str(pseudo_ext) + '\n' + inst_filter, 'output':str(x['std_ans'])})

    train_dataset = PureGenDataset(IFTBuffer(
                        MixIFTGenerator(mydata, data, tokenizer, ratio=0), # ratio=0：不混合通用数据
                        length=args.max_length))

    train(args, model, tokenizer, train_dataset)