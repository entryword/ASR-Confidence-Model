import os, sys
from os import environ, path
import shutil
from pocketsphinx_decoder import decoder
import numpy as np


def preprocess(folder_prefix,tmp_prefix = './.tmpwavs/'):
	try:
		os.mkdir(tmp_prefix)
	except:
		pass

	filenames = [s for s in os.listdir(folder_prefix) if s.endswith('.wav') or s.endswith('.mp3')]
	for filename in filenames:
		print(filename)
		os.system('sox {} -c 1 -r 16000 {}'.format(path.join(folder_prefix,filename),path.join(tmp_prefix,filename)))

def get_asr_result(prefix):
	'''
		get_asr_result(prefix)
		prefix : directory containing .wav
		it would generate .asr at the same directory
	'''
	preprocess(prefix)
	filenames = [s for s in os.listdir('.tmpwavs') if s.endswith('.wav')]
	for filename in filenames:
		decoder.start_utt()
		decoder.process_raw(open('./.tmpwavs/'+filename,'rb').read(100000000),False,False)
		decoder.end_utt()
		seg = list(decoder.seg())
		words = [ x.word for x in seg ]
		feats = [ (x.ascore,x.lscore,x.lback,x.prob)  for x in seg]
		asrname = filename.split('.')[0]+'.asr'
		feats = np.array(feats).mean(axis=0)
		open(path.join(prefix,asrname),'w').write(' '.join(words))
		open(path.join(prefix,asrname+'.feats'),'w').write(' '.join([str(x) for x in feats]))


if __name__=='__main__':
	get_asr_result(sys.argv[1])
