import torch
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Adafactor, AdamW
#from apex import amp
#from apex.parallel import DistributedDataParallel
import sys, os
from itertools import cycle
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
import joblib
import numpy as np
import argparse
import random
from torch.nn.parallel import DataParallel
from tqdm import tqdm, trange

import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    #os.environ['MASTER_ADDR'] = '127.0.0.1'
    #os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=4)
    
   

class PT_dataset(Dataset):
    def __init__(self, pth, max_len=512):
        self.jb = joblib.load(pth)
        self.len = max_len
        

    def __len__(self):
        return len(self.jb[0])

    def __getitem__(self, index):
        inp = self.jb[0][index]
        out = self.jb[1][index]
        inp_ids = torch.zeros(self.len)
        att_msk = torch.zeros(self.len)
        labels = torch.ones(self.len)*(-100)
        inp_ids[:min(self.len, len(inp))] = torch.from_numpy(np.array(inp))[:min(self.len, len(inp))]
        att_msk[:min(self.len, len(inp))] = 1
        labels[:min(self.len, len(out))] = torch.from_numpy(np.array(out))[:min(self.len, len(out))]
        return inp_ids.long(), att_msk.long(), labels.long()

class LM_dataset(Dataset):
    def __init__(self, tokenizer, pth, max_len=512):
        self.tokenizer = tokenizer
        self.data = open(pth).readlines()
        self.len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tok = self.tokenizer.encode(self.data[index].strip('\n'))
        inp = torch.zeros(self.len).long()
        out = torch.ones(self.len).long()*(-100)
        att = torch.ones(self.len).long()
        inp[:min(self.len, len(tok))] = torch.tensor(tok[:min(self.len, len(tok))]).long()
        att[:min(self.len, len(tok))] = 1
        out[0] = 1
        out[1:min(self.len, len(tok)+1)] = torch.tensor(tok[:min(self.len-1, len(tok))]).long()
        return inp, att, out
        
def process_nli(premise: str, hypothesis: str):
    """ process to required mnli format with task prefix """
    return "".join(['mnli premise: ', premise, ' hypothesis: ', hypothesis])

def process_sst(sentencce):
    """ process to required mnli format with task prefix """
    return "".join(['sst2 sentence: ', sentencce])

def do_eval(tokenizer, model, dataloader):
    hit = 0
    total = 0
    model.eval()
    for i in dataloader:
        inputs, atts, labels = i
        out = model.module.generate(input_ids=inputs.cuda(), attention_mask=atts.cuda(), output_scores=True, return_dict_in_generate=True,
                            num_beams=1)
        pos = 0
        scores = out.scores
        while pos<len(scores) and labels[0,pos]>-100:
            predictions = torch.argmax(scores[pos], dim=1)
            hit += (labels[:,pos]==predictions.cpu()).sum().item()
            pos += 1
            total += labels.shape[0]

    return hit/total*100

def get_dataloader(rank, world_size, pth):
    dataset = PT_Dataset(pth)
    sampler = DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    dataloader = DataLoader(
        dataset, batch_size=8, sampler=sampler
    )
    return dataloader



def train(rank, n_procs, args):
    if args.amp:
        args.local_rank = rank
        dist.init_process_group(backend='nccl',
                init_method='tcp://127.0.0.1:29577',#'env://',
                                world_size=n_procs,
                                rank=rank)
        
        print(
            f"{rank + 1}/{n_procs} process initialized.\n"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #device = torch.device('cuda')

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    config = T5Config(
        vocab_size=32128,
        d_model=256,#768
        d_kv=16,#64
        d_ff=768,#3072
        num_layers=4,#12
        num_heads=4,#12
        relative_attention_num_buckets=8,#32
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=1,#32100,
        decoder_start_token_id=1,
        eos_token_id=2,
        gradient_checkpointing=False)

    print('model loading')
    if args.version=='base':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
    elif args.version=='large':
        model = T5ForConditionalGeneration.from_pretrained('t5-large')
    else:
        model = T5ForConditionalGeneration(config)
    if args.amp:
        torch.cuda.set_device(rank)
        model = model.cuda(rank)
        args.batch_size = int(args.batch_size / n_procs)
        optimizer = AdamW(model.parameters(), lr=args.lr)#Adafactor(model.parameters(), lr=None)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
        model = DistributedDataParallel(model)
    else:
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=args.lr)#Adafactor(model.parameters(), lr=None)

    params = []
    for n,p in model.named_parameters():
        params.append(p)

    
    global_step = 0

    accs = []
    rec_loss = 0
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    accu = 256/args.batch_size
    for epoch in range(1):
        print('epoch: ', epoch)
        TrainSet = PT_dataset(args.pth_train)
        train_sampler = RandomSampler(TrainSet)
        if args.amp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(TrainSet)
        train_dataloader = DataLoader(TrainSet, sampler=train_sampler, 
                                      batch_size=args.batch_size, drop_last=True,
                                      num_workers=4, pin_memory=True)
        if args.pth_lm is not None:
            LMSet = LM_dataset(tokenizer, args.pth_lm)
            lm_dataloader = DataLoader(LMSet, shuffle=True,
                                          batch_size=max(1,int(args.batch_size/4)), drop_last=True,
                                          num_workers=4, pin_memory=True)
        
            dataloader_iterator1 = iter(lm_dataloader)
        for idx, data_train in enumerate(train_dataloader):
            if args.pth_lm is not None:
                try:
                    data_lm = next(dataloader_iterator1)
                except Exception as e:
                    dataloader_iterator1 = iter(lm_dataloader)
                    data_lm = next(dataloader_iterator1)
                inputs, atts, labels = data_lm
                if args.amp:
                    preds = model.lm_head(model.decoder(inputs.cuda(rank), atts.cuda(rank))[0])
                else:
                    preds = model.lm_head(model.decoder(inputs.cuda(), atts.cuda())[0])
                loss = loss_func(preds.view(-1, preds.shape[-1]), labels.view(-1).cuda())
                loss = loss/(10*accu)
                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            inputs, atts, labels = data_train
            if args.amp:
                loss = model(input_ids=inputs.cuda(rank), attention_mask=atts.cuda(rank), labels=labels.cuda(rank), return_dict=True)['loss']
            else:
                loss = model(input_ids=inputs.cuda(), attention_mask=atts.cuda(), labels=labels.cuda(), return_dict=True)['loss']
            loss = loss.mean()/accu
            rec_loss += loss.item()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if idx%accu==1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step%10==0 and rank==0:
                    print('step: ', global_step, ', loss: ', rec_loss/10)
                    rec_loss = 0
                # do_eval
                if global_step % 980 == 0 and rank==0:
                    torch.save(model, args.save_pth+str(global_step)+'.pt')

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--version", default='base', type=str,)#base, large, zero
    parser.add_argument("--amp", default=False, type=bool,)
    parser.add_argument("--save_pth", default='save_model/ckpt_basepre', type=str,)
    parser.add_argument("--lr", default=1e-4, type=float,)
    parser.add_argument("--resume", default=-1, type=int,)
    parser.add_argument("--pth_train", default='predata/data_mix.jbl', type=str,)
    parser.add_argument("--pth_lm", default=None, type=str,)
    parser.add_argument("--batch_size", default=8, type=int,)
    parser.add_argument("--epoch", default=5, type=int,)
    parser.add_argument("--seed", default=1234, type=int,)
    parser.add_argument("--local_rank", type=int,)
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.amp:
        args.n_procs = torch.cuda.device_count()
        mp.spawn(train, nprocs=args.n_procs, args=(args.n_procs, args))
    else:
        train(0,0,args)
