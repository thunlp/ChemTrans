from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 

import sys
nm = sys.argv[1]

preds = open(nm+'_pred.txt').readlines()
anss = open(nm+'_ans.txt').readlines()
inps = open(nm+'_inp.txt').readlines()

def Dist(line, N, rec, sumcnt):
    wds = line.strip('\n').lower().split(' ')
    for i in range(len(wds)-N+1):
        tmp = ''
        for tt in range(N):
            tmp += wds[tt+i]
            tmp += ' '
        if tmp not in rec:
            rec[tmp] = 1
    sumcnt += len(wds)-N+1
    return rec, sumcnt

bleu2 = 0
bleu4 = 0
rg2 = 0
rg4 = 0

rec2 = {}
cnt2 = 0
rec4 = {}
cnt4 = 0
rouger = Rouge(metrics=['rouge-2', 'rouge-4'])
for ind in range(len(preds)):
    st1 = preds[ind].strip('\n')
    st2 = gts[ind].strip('\n')
    bleu2 += sentence_bleu([st2.split(' ')], st1.split(' '), weights=[0,1,0,0])
    bleu4 += sentence_bleu([st2.split(' ')], st1.split(' '), weights=[0,0,0,1])
    rg2+=rouger.get_scores(st1, st2)[0]['rouge-2']['f']
    rg4+=rouger.get_scores(st1, st2)[0]['rouge-4']['f']
    rec2, cnt2 = Dist(st1, 2, rec2, cnt2)
    rec4, cnt4 = Dist(st1, 4, rec4, cnt4)
bleu2/=len(preds)
bleu4/=len(preds)
rg2 /= len(preds)
rg4 /= len(preds)
dink2 = len(rec2)/cnt2
dink4 = len(rec4)/cnt4
print(bleu2, bleu4, rg2, rg4, dink2, dink4)
