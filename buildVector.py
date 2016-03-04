import ast
import csv
import numpy as np
from get_answer import prune_asr_result

def makeDictionary():
	global d
	d={}
	dim=200

	with open('/home/ch/DATA/vector') as f:
		for line in f:
			line = line.rstrip('\n')
			cutline = line.split(' ')
			vectors = [float(i) for i in cutline[1:dim+1]]
			d[cutline[0]] = vectors

	f.close()
	return d