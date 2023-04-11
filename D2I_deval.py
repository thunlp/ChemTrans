import sys
nm = sys.argv[1]

preds = open(nm+'_pred.txt').readlines()
anss = open(nm+'_ans.txt').readlines()
inps = open(nm+'_inp.txt').readlines()
from evalchem import evalverb
from nltk.translate.bleu_score import sentence_bleu

sm1 = 0
sm2 = 0
bleu2 = 0
bleu4 = 0
dup = {}
ski = {}
for i in range(len(anss)):
    acc, verbacc, r1, r2, pa1, pa2 = evalverb(' '+preds[i*1].strip('\n'), ' '+anss[i].strip('\n'))
    for wd in r1:
        if wd not in dup:
            dup[wd] = 0
        dup[wd]+=1
    for wd in r2:
        if wd not in ski:
            ski[wd] = 0
        ski[wd]+=1

    sm1+=acc
    sm2+=verbacc
    bleu2 += sentence_bleu([anss[i].split(' ')], preds[i*1].split(' '), weights=[0,1,0,0])
    bleu4 += sentence_bleu([anss[i].split(' ')], preds[i*1].split(' '), weights=[0,0,0,1])
print(dup, ski)
print(bleu2/len(anss), bleu4/len(anss))
print(sm1/len(anss), sm2/len(anss))

