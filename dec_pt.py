import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, Adafactor
import numpy as np
import random
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm import tqdm

class PT_dataset(Dataset):
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

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')

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
    model = model.cuda()
    if args.init_checkpoint is not None:
        pt = torch.load(args.init_checkpoint).state_dict()
        #pt1 = {k[7:]: v for k, v in pt.items()}
        model.load_state_dict(pt, strict=True)
    
    params = []
    for n,p in model.named_parameters():
        params.append(p)

    optimizer = Adafactor(model.parameters(), lr=1e-4, relative_step=False)
    global_step = 0

    TrainSet = PT_dataset(tokenizer, args.pth_train)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler, batch_size=args.batch_size, drop_last=True, num_workers=4, pin_memory=True)
    avg_loss = 0
    avg_cnt = 0
    avg_acc = 0
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    acc = 64/args.batch_size

    for epoch in range(args.epoch):
        print('epoch: ', epoch)
        for idx, i in enumerate(tqdm(train_dataloader)):
            inputs, atts, labels = i
            preds = model.lm_head(model.decoder(inputs.cuda(), atts.cuda())[0])
            loss = loss_func(preds.view(-1, preds.shape[-1]), labels.view(-1).cuda())
            loss = loss/acc
            loss.backward()
            avg_loss += loss.item()
            avg_cnt += 1
            avg_acc += torch.sum(torch.argmax(preds, dim=-1)==labels.cuda()).item()/preds.shape[0]
            if idx%acc==0:
                optimizer.step()
                optimizer.zero_grad()
                if global_step==200 or global_step==500:
                    print('Step: ', global_step, ', acc: ', avg_acc/avg_cnt, ', loss: ', avg_loss/avg_cnt)
                    avg_loss = 0
                    avg_cnt = 0
                    avg_acc = 0
                    torch.save(model, args.save_pth+str(global_step)+'.pt')
                global_step += 1
                if global_step>args.global_step:
                    return

def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--version", default='base', type=str,)#base, large, zero
    parser.add_argument("--init_checkpoint", default=None, type=str,)#'save_model/ckpt_basepre980.pt', type=str,)
    parser.add_argument("--save_pth", default='save_model/ckpt_decbase_', type=str,)
    parser.add_argument("--pth_train", default='data/aug_train.txt', type=str,)
    parser.add_argument("--batch_size", default=16, type=int,)
    parser.add_argument("--epoch", default=5, type=int,)
    parser.add_argument("--seed", default=1234, type=int,)
    parser.add_argument("--global_step", default=501, type=int,)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    main(parse_args())

