import os, sys
from os import environ, path
import shutil
import wave
import re
import pandas as pd
from collections import defaultdict
from operator import attrgetter

from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer
from auditok.cmdline import save_audio_data


def get_all_prefix(folder):
    filenames = set(s for s in os.listdir(folder) if s.endswith('.wav'))
    pat = re.compile('[0-9]+')
    all_prefixs = set(pat.sub('', fname[:-4]) for fname in filenames)
    return all_prefixs

def preprocess(folder,prefix):
    filenames = set(s for s in os.listdir(folder) if s.endswith('.wav'))
    all_prefixs = get_all_prefix(folder)

    cmds = ['sox']
    for i in range(1,10000000):
        fname = '{}{}.wav'.format(prefix,i)
        if fname not in filenames:
            break
        else:
            os.system('sox {} {} channels 1 rate 16000'.format(path.join(folder,fname),path.join('./temp/',fname)))
            cmds.append(path.join('./temp/',fname))


    os.system('mkdir temp')
    cmds.append('./temp/'+prefix+'.wav')
    cmds.append('channels 1 rate 16000')
    print(' '.join(cmds))
    os.system(' '.join(cmds))

def _get_asr_result_whole(folder,prefix):
    asource = ADSFactory.ads(filename='./temp/{}.wav'.format(prefix), block_size=160)
    validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=65)
    tokenizer = StreamTokenizer(validator=validator, min_length=300, max_length=1000, max_continuous_silence=50)
    asource.open()
    from pocketsphinx_decoder import decoder

    tokens = tokenizer.tokenize(asource)


    d = defaultdict(list)


    past = 0
    for content,start,end in tokens:
        save_audio_data(data=b''.join(content), filename='tmp.wav', filetype='wav', sr=asource.get_sampling_rate(),sw = asource.get_sample_width(),ch = asource.get_channels())
        decoder.start_utt()
        decoder.process_raw(open('tmp.wav','rb').read(),False,False)
        decoder.end_utt()
        seg = list(decoder.seg())
        print(' '.join([s.word for s in seg]))
        def add_feature(name,add=None):
            if add is None:
                d[name].extend(list(map(attrgetter(name),seg)))
            else:
                d[name].extend([attrgetter(name)(x)+add for x in seg])
        add_feature('start_frame',past)
        add_feature('end_frame',past)
        add_feature('word')
        add_feature('ascore')
        add_feature('lscore')
        add_feature('lback')
        add_feature('prob')
        past += len(content)
        df = pd.DataFrame(d)
        df = df[['start_frame','end_frame','ascore','lscore','lback','prob','word']]
        df.to_csv(path.join(folder ,'{}.csv'.format(prefix)), index=None)


def get_asr_result(folder):
    all_prefixs = get_all_prefix(folder)

    for prefix in all_prefixs:
        preprocess(folder,prefix)
        _get_asr_result_whole(folder,prefix)




def get_frame_no(time_str):
    h,m,s = [int(x) for x in time_str.split(':')]
    return (h*3600+m*60+s)*100


def read_ans_time(folder,prefix):
    filenames = set(s for s in os.listdir(folder) if s.endswith('.ans'))
    pat = re.compile(' \-\-\> |,')
    ret = [(-1,-1)]

    add = 0
    for i in range(1,10000):
        fname = '{}{}.ans'.format(prefix,i)
        if fname not in filenames:
            break
        else:
            with open(path.join(folder,fname)) as f:
                line = f.readline().strip()
                st,stx,ed,edx = pat.split(line)
                st,ed = get_frame_no(st)+int(stx)/10.0,get_frame_no(ed)+int(edx)/10.0
                if st+add< ret[-1][1]:
                    add = ret[-1][1]
                ret.append((st+add,ed+add))
    return ret[1:]


def gen_asr_from(folder,prefix):
    df = pd.read_csv(path.join(folder,prefix+'.csv'))
    times = read_ans_time(folder,prefix)

    p = 0
    for i,(st,ed) in enumerate(times):
        words = []
        while True:
            if p>=df.shape[0]:
                break
            st_df,ed_df = df.ix[p,['start_frame','end_frame']]
            print(st,ed,st_df,ed_df)
            if ed_df<=st:
                p+=1
                continue
            if st_df>=ed:
                break
            print(df.ix[p,'word'])
            words.append(df.ix[p,'word'])
            p+=1

        fname = '{}{}.asr'.format(prefix,i+1)
        with open(path.join(folder,fname),'w') as f:
            f.write(' '.join(words))




if __name__ == '__main__':
    get_asr_result(sys.argv[1])
