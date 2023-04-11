import torch
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Adafactor, AdamW
import sys, os
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
import joblib
import numpy as np
import argparse
import random
from torch.nn.parallel import DataParallel
from tqdm import tqdm, trange
import torch.nn as nn
import pdb
from evalchem import evalverb
from nltk.translate.bleu_score import sentence_bleu

sav = pickle.load(open('data/token_save.pkl', 'rb'))
rev = {}
for ky in sav.keys():
    rev[sav[ky]] = ky

class QA_dataset(Dataset):
    def __init__(self, tokenizer, pth_i, pth_o, max_len=512, iftrain=False, few=1, mutual=0):
        self.data0 = (open(pth_i)).readlines()
        self.ans0 = (open(pth_o)).readlines()
        if iftrain:
            seq = np.arange(len(self.data0))
            np.random.shuffle(seq)
            self.data = [self.data0[i] for i in seq[:int(len(seq)*few)]]
            self.ans = [self.ans0[i] for i in seq[:int(len(seq)*few)]]
        else:
            self.data = self.data0
            self.ans = self.ans0
        self.len = max_len
        self.tokenizer = tokenizer
        self.mutual = mutual
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tmpinp = self.data[index].strip('\n')
        if self.mutual:
            tmpinp = self.ans[index].strip('\n')
        for ky in sav.keys():
            tmpinp = tmpinp.replace(ky, sav[ky])
        tmpout = self.ans[index].strip('\n')
        if self.mutual:
            tmpout = self.data[index].strip('\n')
        for ky in sav.keys():
            tmpout = tmpout.replace(ky, sav[ky])
        inp = self.tokenizer.encode(tmpinp+' . Operation:')
        out = self.tokenizer.encode(tmpout)
        inp_ids = torch.ones(self.len)
        att_msk = torch.zeros(self.len)
        labels = torch.ones(self.len)*(-100)
        inp_ids[:min(self.len, len(inp))] = torch.from_numpy(np.array(inp))[:min(self.len, len(inp))]
        att_msk[:min(self.len, len(inp))] = 1
        labels[:min(self.len, len(out))] = torch.from_numpy(np.array(out))[:min(self.len, len(out))]

        return inp_ids.long(), att_msk.long(), labels.long()

def do_eval(tokenizer, model, dev_dataloader, pth=None, beam=1, mutual=0):
    cnt = 0
    f1 = open(pth+'inp.txt', 'w')
    f2 = open(pth+'pred.txt', 'w')
    f3 = open(pth+'ans.txt', 'w')
    bleu2 = 0
    bleu4 = 0
    sumacc = 0
    sumverb = 0
    with torch.no_grad():
        for idx, i in enumerate(tqdm(dev_dataloader)):
            inputs, atts, labels = i
            output = model.generate(input_ids=inputs.cuda(), attention_mask=atts.cuda(), no_repeat_ngram_size=0, repetition_penalty=1, num_beams=beam, max_length=512)#, num_return_sequences=5)#, top_p=0.9, do_sample = True)
            tmp = (labels>0)*labels+(labels<0)
            pred = [tokenizer.decode(piece[1:]).split('</s>')[0].strip('<pad> ') for piece in output]
            ans = [tokenizer.decode(piece).split('</s>')[0] for piece in tmp]
            inp = [tokenizer.decode(piece).split('</s>')[0] for piece in inputs]
            for i in range(len(pred)):
                for ky in rev.keys():
                    pred[i] = pred[i].replace(ky, rev[ky])
                    ans[i] = ans[i].replace(ky, rev[ky])
                    inp[i] = inp[i].replace(ky, rev[ky])
            for st in inp:
                f1.write(st+'\n')
                f1.flush()
            for st in pred:
                f2.write(st+'\n')
                f2.flush()
            for st in ans:
                f3.write(st+'\n')
                f3.flush()
            
            for one in range(len(ans)):
                st1 = ' '+pred[one]
                st2 = ' '+ans[one]
                if mutual:
                    bleu2 += sentence_bleu([st2.split(' ')], st1.split(' '), weights=[0,1,0,0])
                    bleu4 += sentence_bleu([st2.split(' ')], st1.split(' '), weights=[0,0,0,1])
                else:
                    acc, verbacc, r1, r2, pa1, pa2 = evalverb(st1, st2)
                    sumacc += acc
                    sumverb += verbacc
                cnt += 1
            if mutual:
                print(bleu2/cnt, bleu4/cnt)
            else:
                print(sumacc/cnt, sumverb/cnt)
    if mutual:
        return bleu2/cnt, bleu4/cnt
    return sumacc/cnt, sumverb/cnt

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    print('tokenizer loading')
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

    print('ckpt loading')
    
    if args.init_checkpoint is not None:
        pt = torch.load(args.init_checkpoint).state_dict()#['state_dict'] 
        #pt1 = {k[7:]: v for k, v in pt.items()}
        model.load_state_dict(pt, strict=True)
    

    model = model.cuda()
    params = []
    for n,p in model.named_parameters():
        params.append(p)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    #optimizer = Adafactor(model.parameters())#, scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
       
    global_step = 0

    print('dataset loading')
    TrainSet = QA_dataset(tokenizer, args.pth_train+'inp.txt', args.pth_train+'out.txt', iftrain=True, few=args.few, mutual=args.mutual)
    DevSet = QA_dataset(tokenizer, args.pth_dev+'inp.txt', args.pth_dev+'out.txt', mutual=args.mutual)
    TestSet = QA_dataset(tokenizer, args.pth_test+'inp.txt', args.pth_test+'out.txt', mutual=args.mutual)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=4, pin_memory=True)
    
    dev_dataloader = DataLoader(DevSet,
                                  batch_size=args.batch_size*4, drop_last=False,
                                  num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(TestSet,
                                  batch_size=args.batch_size*2, drop_last=False,
                                  num_workers=4, pin_memory=True)
    avg_loss = 0
    best_acc = 0
    tag = 0
    rc = []
    print('start training')
    accu=16/args.batch_size
    for epoch in range(args.epoch):
        if tag:
            break
        print('epoch: ', epoch)
        for idx, i in enumerate(tqdm(train_dataloader)):
            if tag:
                break
            inputs, atts, labels = i
            output = model(input_ids=inputs.cuda(), attention_mask=atts.cuda(), labels=labels.cuda(), return_dict=True)
            loss = output.loss.mean()/accu
            loss.backward()
            avg_loss += loss.item()
            if idx%accu==0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step%25==0:#25
                    print('Step: ', global_step, ', loss: ', avg_loss)
                    avg_loss = 0

                if global_step>args.global_step:
                    return
                # do_eval
                if global_step % int(175*args.few) == 0:#175
                    if global_step>args.step_pre:
                        acc, vacc = do_eval(tokenizer, model, dev_dataloader, args.log_pth, mutual=args.mutual)
                        if args.mutual:
                            print('Step:', global_step, ', bleu2:', acc, ', bleu4:', vacc)
                        else:
                            print('Step:', global_step, ' , acc:', acc, ', verb acc:', vacc)
                        print(rc)
                        if len(rc)>2:
                            if acc<rc[-3] and rc[-1]<rc[-3] and rc[-2]<rc[-3]:
                                tag = 1
                                break
                        rc.append(acc)
                        if acc>=best_acc:
                            best_acc = acc
                            torch.save(model, args.save_pth)
    acc, vacc = do_eval(tokenizer, model, dev_dataloader, args.log_pth)
    print('Step:', global_step, ' , acc:', acc, ', verb acc:', vacc)
    if acc>=best_acc:
        best_acc = acc
        torch.save(model, args.save_pth)
    
    model.load_state_dict(torch.load(args.save_pth).state_dict())
    acc,vacc = do_eval(tokenizer, model, test_dataloader, args.log_pth, beam=3, mutual=args.mutual)
    print(acc, vacc)

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--init_checkpoint", default=None, type=str,)
    parser.add_argument("--version", default='base', type=str,)#base, large, zero
    parser.add_argument("--mutual", default=0, type=str,)#0 for D2I and 1 for I2D
    parser.add_argument("--save_pth", default='save_model/finetune_decbase1.pt', type=str,)
    parser.add_argument("--log_pth", default='log/ft_decbase1_', type=str,)
    parser.add_argument("--lr", default=1e-4, type=float,)#1e-4
    parser.add_argument("--resume", default=-1, type=int,)
    parser.add_argument("--pth_train", default='data/train_', type=str,)
    parser.add_argument("--pth_dev", default='data/dev_', type=str,)
    parser.add_argument("--pth_test", default='data/test_', type=str,)
    parser.add_argument("--batch_size", default=8, type=int,)#16
    parser.add_argument("--epoch", default=20, type=int,)#20
    parser.add_argument("--seed", default=1111, type=int,)
    parser.add_argument("--global_step", default=100000, type=int,)#1000
    parser.add_argument("--step_pre", default=800, type=int,)#300
    parser.add_argument("--few", default=1, type=float,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())

