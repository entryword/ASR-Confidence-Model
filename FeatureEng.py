import os, sys
import numpy as np
import getopt
from sklearn import linear_model
from sklearn import tree
from sklearn import metrics

import BaseFeature
import FeatureGenerator
import Label
import computeVector
import StableASRFeature

Features = {'base' : (BaseFeature.BaseFeature, 'base.csv'), 
			'noise' : (FeatureGenerator.feaname, 'noise.csv'), 
			'ans' : (Label.Label, 'acc.csv', 'edis.csv'), 
			'lang' : (computeVector.vectorFeature, 'lang.csv'),
			'asr' : (StableASRFeature.ASRFeature, 'asr.csv')
			}

def main(argc:int, argv:list):
	SrcDir = 'Data/Source/'
	DestDir = ''
	Feature = 'all'
	Prefix = 'ted'

	try:
		Params, args = getopt.getopt(argv[1:], 's:d:f:p:')
	except getopt.GetoptError:
		print("Invalid Param")
		sys.exit(2)

	for Cmd, Val in Params:
		if Cmd == '-s':
			SrcDir = Val
		if Cmd == '-d':
			DestDir = Val
		if Cmd == '-f':
			Feature = Val
		if Cmd == '-p':
			Prefix = Val

	if DestDir == '':
		DestDir = 'Data/' + Prefix + '/'

	if not os.path.exists(DestDir):
		os.makedirs(DestDir)

	print("SrcDir: ", SrcDir)
	print("DestDir: ", DestDir)
	print("Feature: ", Feature)
	print("Prefix: ", Prefix)
	
	FeaToUse = []
	for f in Features:
		if f == Feature or Feature == 'all':
			FeaToUse.append(f)

	rootdir = os.getcwd()
	for f in FeaToUse:
		print('Parsing ' + Features[f][0].__name__)
		files = []
		for n in Features[f][1:]:
			files.append(open(DestDir + n, 'w'))
		os.chdir(SrcDir)
		Features[f][0](Prefix, *files)
		for n in files:
			n.close()
		os.chdir(rootdir)

if __name__ == "__main__":
	main(len(sys.argv), sys.argv)