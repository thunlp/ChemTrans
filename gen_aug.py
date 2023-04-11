oper_name = {0: 'add', 1: 'wash', 2: 'extract', 3: 'dry', 4: 'filter', 5: 'column', 6: 'distill', 7: 'evaporate', 8: 'settemp', 9: 'reflux', 10: 'quench', 11: 'triturate', 12: 'recrystallize', 13: 'partition', 14: 'transfer', 15: 'yield'}
rec_oper = {8: ['temperature(OPERATION)', 'N/A', 'time(OPERATION)', 'pressure', 'temperature(reagent)', 'reagent', 'speed(reagent)', 'type:pure', 'note'], 15: ['N/A', 'appearance', 'yield(OPERATION)', 'mass(yield)', 'reagent', 'mass(reagent)', 'note', 'mole', 'type:pure', 'concentration', 'temperature(OPERATION)', 'name', 'pressure', 'time(OPERATION)', 'volume'], 0: ['N/A', 'reagent', 'temperature(OPERATION)', 'temperature(reagent)', 'time(OPERATION)', 'batch:total', 'speed(reagent)', 'name', 'volume', 'type:mixture', 'type:pure', 'mole', 'note', 'mass(reagent)', 'concentration', 'phase', 'pressure', 'reagent1', 'UNC', 'appearance', 'mass(yield)'], 4: ['N/A', 'reagent', 'phase', 'UNC', 'pressure', 'batch:total', 'note'], 1: ['reagent', 'phase', 'speed(reagent)', 'N/A', 'batch:each', 'volume', 'name'], 14: ['N/A', 'reagent1', 'reagent(transfer)', 'reagent(target)', 'temperature(OPERATION)', 'batch:total', 'speed(reagent)', 'UNC', 'reagent', 'reagent2', 'note', 'name', 'batch:each'], 9: ['time(OPERATION)', 'temperature(OPERATION)', 'reagent', 'N/A'], 6: ['N/A', 'pressure', 'temperature(OPERATION)', 'temperature(reagent)', 'time(OPERATION)', 'reagent', 'speed(reagent)', 'UNC', 'phase'], 7: ['N/A', 'pressure', 'temperature(OPERATION)', 'time(OPERATION)', 'UNC', 'batch:each', 'type:mixture', 'reagent'], 12: ['reagent', 'N/A', 'appearance', 'type:pure', 'temperature(OPERATION)'], 3: ['reagent', 'pressure', 'N/A', 'time(OPERATION)', 'phase', 'batch:each', 'type:pure', 'temperature(OPERATION)'], 2: ['reagent', 'phase', 'N/A', 'time(OPERATION)'], 10: ['reagent', 'N/A', 'temperature(OPERATION)', 'time(OPERATION)'], 5: ['reagent(adsorbent)', 'reagent(eluent)', 'N/A', 'reagent(target)', 'reagent'], 11: ['reagent', 'N/A', 'temperature(OPERATION)'], 13: ['reagent', 'N/A'], 'stirring': ['reagent'], '300 ml. of water.': [], '51.0 mL (0.400 mol) of freshly distilled chlorotrimethylsilane.': [], '1.51 g (3.00 mmol) of (phenyl)[2-(trimethylsilyl)phenyl]iodonium triflate (2)': ['reagent'], 'aqueous phase': ['reagent'], 'washed': [], 'dissolved': ['temperature(OPERATION)'], 'filtered': ['reagent'], 'about 500 cc. of the lighter liquid': [], 'added': ['reagent', 'temperature(OPERATION)', 'speed(reagent)'], '(-20 °C': ['time(OPERATION)', 'temperature(OPERATION)'], 'aqueous layer': [], 'the aqueous layer': ['reagent']}
rec_reag = ['name', 'type', 'volume', 'mole', 'mass(reagent)', 'concentration', 'note', 'speed(reagent)', 'batch:each', 'batch:total', 'temperature(reagent)', 'mass(yield)', 'yield(OPERATION)', 'UNC', 'time(OPERATION)', 'equivalent', 'phase', 'temperature(OPERATION)', 'appearance', 'N/A', 'reagent1', 'reagent2']


f = open('data/allout0630.txt')
lines = f.readlines()[-2000:]
f.close()

rec = {}

import random

def findpara(sent):
    pos = sent.find(':')
    while pos>-1:
        pos1 = pos-1
        while sent[pos1]!=' ':
            pos1-=1
        pos2 = pos+1
        while sent[pos2]!='&':
            pos2+=1
        typ = sent[pos1+1:pos].strip(' ')
        para = sent[pos+1:pos2].strip(' ')
        if typ not in rec:
            rec[typ] = []
        if para not in rec[typ]:
            rec[typ].append(para)
        pos = sent.find(':', pos+1)
    return

def getreag():
    sent = 'reagent: ( '
    for ky in rec_reag:
        rd = random.randint(0,2)
        if rd==0 and ky in rec:
            sent += ky+': '
            rk = random.randint(0, len(rec[ky])-1)
            sent += rec[ky][rk]+' & '
    sent += ') '
    return sent

def getoper():
    rd = random.randint(0,15)
    oper = oper_name[rd]
    sent = '[ '+oper+' ] '
    pars = rec_oper[rd]
    for par in pars:
        rd = random.randint(0,2)
        if rd==0:
            if par=='reagent':
                st = getreag()
            else:
                st = ''
                if par in rec:
                    rk = random.randint(0, len(rec[par])-1)
                    st = par+': '+rec[par][rk]+' & '
            sent += st
    return sent


for line in lines:
    findpara(line.strip('\n'))

import pickle
pickle.dump(rec, open('pararec.pkl', 'wb'))

fw = open('data/aug_train.txt', 'w')

for i in range(10000):
    rd = random.randint(3,10)
    sent = ''
    for j in range(rd):
        sent += getoper()
    fw.write(sent+'\n')

fw.close()


