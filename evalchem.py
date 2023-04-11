import numpy as np
#from fuzzywuzzy import fuzz
import difflib
from nltk.translate.bleu_score import sentence_bleu


def dynamic(seq1, seq2, pa1=None, pa2=None):
    score = np.zeros((len(seq1), len(seq2)))
    vsc = np.zeros((len(seq1), len(seq2)))
    rec = np.zeros((len(seq1), len(seq2)))
    for i in range(score.shape[0]):
        if seq1[i]==seq2[0]:
            if pa1 is None:
                score[i,0] = 1
            else:
                score[i,0] = evalpara(pa1[i], pa2[0])
                vsc[i,0] = 1
            rec[i,0] = 1
    for j in range(score.shape[1]):
        if seq2[j]==seq1[0]:
            if pa1 is None:
                score[0,j] = 1
            else:
                score[0,j] = evalpara(pa1[0], pa2[j])
                vsc[0,j] = 1
    for i in range(1,score.shape[0]):
        for j in range(1,score.shape[1]):
            tmp1 = score[i-1,j-1]
            vp1 = vsc[i-1,j-1]
            if seq1[i]==seq2[j]:
                if pa1 is None:
                    tmp1 = score[i-1,j-1]+1
                else:
                    tmp1 = score[i-1,j-1]+evalpara(pa1[i], pa2[j])
                    vp1 = vsc[i-1,j-1]+1
            tmp2 = score[i-1,j]
            tmp3 = score[i,j-1]
            vp2 = vsc[i-1,j]
            vp3 = vsc[i,j-1]
            score[i,j] = max(max(tmp2,tmp3), tmp1)
            vsc[i,j] = max(max(vp2, vp3), vp1)
            if score[i,j]==tmp1:
                rec[i,j] = 2
            elif score[i,j]==tmp2:
                rec[i,j] = 1
    mx = score[len(seq1)-1, len(seq2)-1]
    vx = vsc[len(seq1)-1, len(seq2)-1]
    r1 = []
    r2 = []
    i = 0
    j = 0
    low = []
    while i<len(seq1) and j<len(seq2):
        tmp = rec[len(seq1)-i-1, len(seq2)-j-1]
        if seq1[len(seq1)-i-1]==seq2[len(seq2)-j-1]:
            if score[len(seq1)-i-1, len(seq2)-j-1]<0.6:
                low.append(seq1[len(seq1)-i-1])
            r1.append(len(seq1)-i-1)
            r2.append(len(seq2)-j-1)
        if tmp==2:
            i+=1
            j+=1
        elif tmp==1:
            i+=1
        else:
            j+=1
    return r1, r2, mx, vx, low




def evalverb(st1, st2):
    vb1 = []
    pos = st1.find(' [ ')
    pa1 = []
    while pos>-1:
        ps2 = st1.find(' ] ',pos)
        vb = st1[pos+1:ps2+2]
        vb1.append(vb)
        pos = st1.find(' [ ', pos+1)
        if pos>-1:
            pa1.append(st1[ps2+3:pos])
        else:
            pa1.append(st1[ps2+3:])
    vb2 = []
    pos = st2.find(' [ ')
    pa2 = []
    while pos>-1:
        ps2 = st2.find(' ] ',pos)
        vb = st2[pos+1:ps2+2]
        vb2.append(vb)
        pos = st2.find(' [ ', pos+1)
        if pos>-1:
            pa2.append(st2[ps2+3:pos])
        else:
            pa2.append(st2[ps2+3:])
    if len(vb1)==0 or len(vb2)==0:
        return 0, 0, [], [], [], []
    r1,r2,mx,vx, low = dynamic(vb1, vb2, pa1, pa2)
    dup = []
    ski = []
    for i in range(len(vb1)):
        if i not in r1:
            ski.append(vb1[i])
    for i in range(len(vb2)):
        if i not in r2:
            dup.append(vb2[i])
    return 2*mx/(len(vb1)+len(vb2)), 2*vx/(len(vb1)+len(vb2)), low, ski, pa1, pa2

def evalpara(st1, st2):
    pa1 = []
    pos = st1.find(':')
    if st1[pos-4:pos]=='gent':
        pos = st1.find(':', pos+1)
    while pos>-1:
        tmpi = pos
        while tmpi>0:
            tmpi-=1
            if st1[tmpi]==' ':
                break
        ps2 = st1.find('&', pos)
        if len(st1[pos:ps2].strip(' '))>=3:
            pa1.append(st1[tmpi:ps2])
        pos = st1.find(':', pos+1)
        if pos>3 and st1[pos-4:pos]=='gent':
            pos = st1.find(':', pos+1)
    pa2 = []
    pos = st2.find(':')
    if st2[pos-4:pos]=='gent':
        pos = st2.find(':', pos+1)
    while pos>-1:
        tmpi = pos
        while tmpi>0:
            tmpi-=1
            if st2[tmpi]==' ':
                break
        ps2 = st2.find('&', pos)
        if len(st2[pos:ps2].strip(' '))>=3:
            pa2.append(st2[tmpi:ps2])
        pos = st2.find(':', pos+1)
        if pos>3 and st2[pos-4:pos]=='gent':
            pos = st2.find(':', pos+1)
    pa1 = set(pa1)
    pa2 = set(pa2)
    right = 0#[]
    lt1 = len(pa1)
    lt2 = len(pa2)
    if lt1==0 or lt2==0:
        return 0
    for item in pa1:
        mx = 0
        rec = ''
        for tit in pa2:
            '''
            if difflib.SequenceMatcher(None, item, tit).ratio()*100>mx:
            #if fuzz.partial_ratio(item, tit)>mx:
                mx = difflib.SequenceMatcher(None, item, tit).ratio()*100
                #mx = fuzz.partial_ratio(item, tit)
                rec = tit
            '''
            bleu = 100*sentence_bleu([[ w for w in item]], [w for w in tit], weights=[0.25,0.25,0.25,0.25])
            if bleu>mx:
                mx = bleu
                rec = tit
        if mx>=80:#95
            right += mx
            #right.append(rec)
            pa2.remove(rec)
    '''
    dp = len(right)/lt1
    dr = len(right)/lt2
    df = 2*dp*dr/(dp+dr+1e-5)
    return df
    '''
    return right/(50*(lt1+lt2))

'''
f = open('pred_dia0.txt')
preds = f.readlines()
f.close()
f = open('ans_dia0.txt')
gts = f.readlines()
f.close()

allacc = 0
import pdb
pdb.set_trace()
for i in range(len(preds)):
    pred = ' '+preds[i].strip('\n')
    gt = ' '+gts[i].strip('\n')
    acc, r1, r2, pa1, pa2 = evalverb(pred,gt)
    allacc+=acc

print(allacc/len(preds))
'''
