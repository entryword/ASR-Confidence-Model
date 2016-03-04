import numpy as np


def get_asr_feature(prefix,fout):
    df = pd.read_csv(prefix+'.csv')
    times = read_ans_time(folder,prefix)
    p = 0
    for i,(st,ed) in enumerate(times):
        feats = []
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
            words.append([df.ix[p,2:5]])
            p+=1

	feats = np.array(feats)
	feats = feats.mean(axis=0)
        fout.write(' '.join(map(str,feats)+'\n')
