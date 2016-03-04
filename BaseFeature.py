import os
import FeatureGenerator
import wave
import math
import contextlib

import get_answer

def GetWavLen(Path:str):
	with contextlib.closing(wave.open(Path,'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		return math.floor(frames / float(rate))

def GetCharCount(Path:str):
	with open(Path) as fin:
		return len(get_answer.prune_asr_result(fin.read()))

def BaseFeature(Prefix:str, Fout):
	i = 1
	while (True):
		wav = Prefix + str(i) + '.wav'
		asr = Prefix + str(i) + '.asr'
		print('\r' + str(i))
		# you can assume anything is in the working directory
		if os.path.isfile(wav) and os.path.isfile(asr):
			Fout.write(str(GetCharCount(asr)) + ', ' + str(GetWavLen(wav)) + '\n')
		else:
			break
		i += 1
	print('')
	return 0
