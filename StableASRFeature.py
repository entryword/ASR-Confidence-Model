import os

def ASRFeature(Prefix:str, Fout):
	i = 1
	while (True):
		wav = Prefix + str(i) + '.wav'
		asr = Prefix + str(i) + '.asr'
		asrf = Prefix + str(i) + '.asr.feats'
		print('\r' + str(i))
		# you can assume anything is in the working directory
		if os.path.isfile(wav) and os.path.isfile(asr):
			with open(asrf) as fin:
				line = fin.read().strip(' \n')
				tokens = line.split()
				Fout.write(', '.join(tokens) + '\n')
		else:
			break
		i += 1
	print('')
	return 0
