import torch
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Adafactor, AdamW
import sys, os

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
import pickle

sav = pickle.load(open('data/token_save.pkl', 'rb'))
rev = {}
for ky in sav.keys():
    rev[sav[ky]] = ky

if_cuda = True
if_mutual = 0 #0 for D2I and 1 for I2D
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')

pth = 'save_model/finetune_declargepre1.pt'
if if_mutual:
    pth = 'save_model/finetunerev_aug_largepre.pt'

if if_cuda:
    model.load_state_dict(torch.load(pth).state_dict())
    model = model.cuda()
else:
    model.load_state_dict(torch.load(pth, map_location=torch.device('cpu') ).state_dict())
model.eval()

oper_dic = {1:'add', 2:'settemp', 3:'yield', 4:'wash', 5:'filter', 6:'evaporate', 7:'dry', 8:'distill', 9:'extract', 10:'transfer', 11:'reflux', 12:'recrystallize', 13:'quench', 14:'column', 15:'triturate', 16:'partition'}
augs_dic = []
augs_dic.append({1:'name', 2:'type', 3:'mass', 4:'volume', 5:'speed', 6:'concentration', 7:'equivalent', 8:'batch', 9:'note', 10:'temperature', 11:'mole'})
augs_dic.append({1:'temperature', 2:'reagent'})
augs_dic.append({1:'time', 2:'temperature'})
augs_dic.append({1:'reagent name', 2:'appearance', 3:'mass(yield)', 4:'yield'})
augs_dic.append({1:'reagent', 2:'phase'})
augs_dic.append({1:'reagent', 2:'phase'})
augs_dic.append({1:'temperature', 2:'pressure'})
augs_dic.append({1:'reagent'})
augs_dic.append({1:'temperature', 2:'pressure'})
augs_dic.append({1:'reagent', 2:'phase'})
augs_dic.append({1:'reagent1', 2:'reagent2', 3:'temperature'})
augs_dic.append({1:'time'})
augs_dic.append({1:'reagent'})
augs_dic.append({1:'temperature', 2:'reagent'})
augs_dic.append({1:'adsorbent', 2:'eluent'})
augs_dic.append({1:'temperature', 2:'reagent'})
augs_dic.append({1:'reagent1', 2:'reagent2'})

if if_mutual:
    while True:
        print(oper_dic)
        txt = input('choose operation (17 for stop): ')
        inp = ''
        while txt:
            try:
                oper = int(txt)
            except Exception as e:
                oper = 17
            if oper in oper_dic:
                inp += ' [ '+oper_dic[oper]+' ] '
                tmpdic = augs_dic[oper]
                print(tmpdic)
                augs = input('choose augments (0 for stop): ')
                while augs:
                    try: 
                        aug = int(augs)
                    except Exception as e:
                        aug = 0
                    if aug in tmpdic:
                        if tmpdic[aug][:7]!='reagent':
                            inp += tmpdic[aug]+': '
                            val = input('value for '+tmpdic[aug]+':')
                            inp += val+' & '
                        else:
                            inp += 'reagent: ( '
                            print(augs_dic[0])
                            reas = input('choose reagent augments (0 for stop): ')
                            while reas:
                                try:
                                    rea = int(reas)
                                except Exception as e:
                                    rea = 0
                                if rea in augs_dic[0]:
                                    inp += augs_dic[0][rea]+': '
                                    val = input('value for '+augs_dic[0][rea]+':')
                                    inp += val+' & '
                                elif rea==0:
                                    inp += ') '
                                    break
                                reas = input('choose reagent augments (0 for stop): ')
                    elif aug==0:
                        break
                    augs = input('choose augments (0 for stop): ')
            elif oper==17:
                break
            txt = input('choose operation (17 for stop): ')
        print('Your input instructions are: '+inp)
        inp = input('verify your input:')
        for ky in sav.keys():
            inp = inp.replace(ky, sav[ky])
        tok = torch.tensor(tokenizer.encode(inp+'. Operation: ')[:512]).long().unsqueeze(0)
        att = torch.ones(tok.shape).long()
        if if_cuda:
            tok = tok.cuda()
            att = att.cuda()

        output = model.generate(input_ids=tok, attention_mask=att, no_repeat_ngram_size=0, repetition_penalty=1, num_beams=3, max_length=512)
        pred = tokenizer.decode(output[0][1:]).split('</s>')[0].strip('<pad> ')
        for ky in rev.keys():
            pred = pred.replace(ky, rev[ky])
        print(pred)
        print('\n')


else:
    while True:
        txt = input('SI input: ')
        for ky in sav.keys():
            txt = txt.replace(ky, sav[ky])
        inp = torch.tensor(tokenizer.encode(txt+'. Operation: ')[:512]).long().unsqueeze(0)
        att = torch.ones(inp.shape).long()
        if if_cuda:
            inp = inp.cuda()
            att = att.cuda()

        output = model.generate(input_ids=inp, attention_mask=att, no_repeat_ngram_size=0, repetition_penalty=1, num_beams=3, max_length=512)
        pred = tokenizer.decode(output[0][1:]).split('</s>')[0].strip('<pad> ')
        for ky in rev.keys():
            pred = pred.replace(ky, rev[ky])
        print(pred)
        print('\n')

