import re
#import pandas as pd
import os
from os import path
import string
from editdistance import editdistance

def get_id(s):
    st,ed = re.search('[0-9]+\.',s).span()
    return int(s[st:ed-1])

def get_sentence(fname):
    try:
        f = open(fname,'r')
        lines = f.readlines()


        if len(lines)==1:
            #asr file
            return lines[0].strip()
        else:
            #ans file
            return lines[1].strip()
    except FileNotFoundError:
        return ''

def prune_asr_result(asr_sentence):
    asr_sentence_split = asr_sentence.split()
    banned_list = '\[|\]|<|>|\(|\)|--'
    exclude = set(string.punctuation)
    exclude.remove('-')
    remove_braces = lambda s:re.sub('\(.*\)','',s)
    remove_punct = lambda s: ''.join(ch for ch in s if ch not in exclude)


    return ' '.join(remove_punct(remove_braces(word)).lower() for word in asr_sentence_split if not re.match(banned_list,word))

def get_anwser(folder,name):
    filelist = os.listdir(folder)
    asr_list = [fname for fname in filelist if fname.endswith('.asr') and name in fname]
    asr_list.sort(key = get_id)
    asr_list.insert(0,'')
    asr_list.append('')

    c= 0

    ret = []
    for past,now,future in zip(asr_list,asr_list[1:],asr_list[2:]):
        sa= get_sentence(path.join(folder,now))

        st1 = get_sentence(path.join(folder,past[:-4]+'.ans'))
        st2 = get_sentence(path.join(folder,now[:-4]+'.ans'))
        st3 = get_sentence(path.join(folder,future[:-4]+'.ans'))

        st = ' '.join([st1,st2,st3])


        sa = prune_asr_result(sa)
        st = prune_asr_result(st)
        d = (editdistance(sa.split()[1:-1],st.split())- len(st.split()) + len(sa.split()[1:-1]))
        ret.append((d,d==0))


    return ret






