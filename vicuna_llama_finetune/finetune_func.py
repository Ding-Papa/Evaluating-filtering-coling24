import os, sys, argparse, json, time, random
import torch
import deepspeed
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers

from utils import flash_attn_forward, flash_attn_prepare_decoder_attention_mask

from config import lora_config, DS_CONFIG_lora, DS_CONFIG_ft

def absjoin(*args): return os.path.abspath(os.path.join(*args))

def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = flash_attn_prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flash_attn_forward


def get_model_layers(model):
    layers = [["", model]]
    i = 0
    while i < len(layers):
        for nc, lc in layers[i][1].named_children():
            layers.append([f"{layers[i][0]}.{nc}" if layers[i][0] else nc, lc])
        i += 1
    return layers

def prepare(args):
    global DS_CONFIG, device
    if args.use_flash_attention:
        print('using flash attn!!')
        replace_llama_attn_with_flash_attn()
    else:
        print('not using flash attn!!')
    DS_CONFIG = DS_CONFIG_lora if args.use_lora else DS_CONFIG_ft
    print(DS_CONFIG)
    device = torch.device("cuda")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()

    if torch.distributed.get_rank() == 0:
        for path in [args.save_dir, args.save_dir + args.save_name]:
            if not os.path.exists(path): os.mkdir(path)


def train(args, model, tokenizer, train_dataset):
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        sampler=train_sampler,
        batch_size=DS_CONFIG["train_micro_batch_size_per_gpu"]
    )

    engine, _, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model, 
        model_parameters=model.parameters(),
    )
    print("model loaded.")

    epoch_steps = len(train_dataloader) if args.epoch_steps <= 0 else args.epoch_steps
    args.max_steps = args.max_epoches * epoch_steps

    global_step = 0
    engine.train()
    for epoch in range(args.max_epoches):
        losses = []
        if torch.distributed.get_rank() != -1: train_sampler.set_epoch(epoch)
        if torch.distributed.get_rank() == 0:
            pbar = tqdm(range(epoch_steps))

        for batch in train_dataloader:
            #pdb.set_trace()
            if type(batch) == torch.Tensor: xx = batch; yy = batch
            elif len(batch) == 2: xx, yy = batch[0], batch[1]
            loss = engine(
                input_ids = xx.to(device),
                labels = yy.to(device),
                use_cache=False
            ).loss

            engine.backward(loss)
            engine.step()
            # engine.zero_grad()

            global_step += 1
            losses.append(loss.item())
            if args.save_steps > 10 and global_step % args.save_steps == 0:
                dist.barrier()
                if torch.distributed.get_rank() == 0:
                    save_name = f"{args.save_dir + args.save_name + '/' + args.save_name}_epoch{epoch}"
                    if args.use_lora: model.save_pretrained(save_name)
                    else: engine.save_pretrained(save_name)
                    tokenizer.save_pretrained(save_name)
                dist.barrier()

            if torch.distributed.get_rank() == 0:
                pbar.update()
                avgloss = sum(losses[-200:]) / len(losses[-200:])
                pbar.set_description(f"loss: {avgloss:.6f}")

            if global_step >= args.max_steps: break
            if len(losses) >= epoch_steps: break

        dist.barrier()
        if torch.distributed.get_rank() == 0:
            save_name = f"{args.save_dir + args.save_name + '/' + args.save_name}_epoch{epoch}"
            if args.use_lora: model.save_pretrained(save_name)
            else: engine.save_pretrained(save_name)
            tokenizer.save_pretrained(save_name)
        dist.barrier()

        if torch.distributed.get_rank() == 0: pbar.close()
        if global_step >= args.max_steps: break

