import os, sys
import numpy as np
import getopt
import itertools
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn import tree
from sklearn import metrics
from sklearn import cross_validation
from sklearn import svm
from sklearn import ensemble
from sklearn import neural_network
from sklearn import neighbors
from sklearn import base
from sklearn import preprocessing
from sklearn import naive_bayes

def ParseParam(Param):
	Cmd = []
	Val = []
	if(not isinstance(Param, list)):
		Param = Param.split()
	for str in Param:
		str = str.strip()
		if (str != ''):
			if(len(Cmd) > len(Val)):
				if(str.isdigit()):
					if('.' in str):
						Val.append(float(str))
					else:
						if(len(str) == 1 or str[0] != '0'):
							Val.append(int(str))
						else:
							Val.append(str)
				else:
					Val.append(str)
			else:
				Cmd.append(str.strip('-'))
	return Cmd, Val

def SetParam(Model, Param):
	Cmd, Val = ParseParam(Param)
	if(len(Cmd) != len(Val)):
		raise exceptions.SyntaxError('Invalid Parameters')
	for i in range(len(Cmd)):
		if(hasattr(Model, Cmd[i])):
			if(isinstance(getattr(Model, Cmd[i]), ( str ) )):
				setattr(Model, Cmd[i], Val[i])
			else:
				setattr(Model, Cmd[i], float(Val[i]))
	return Model

def GetParamDict(Param):
	Cmd, Val = ParseParam(Param)
	if(len(Cmd) != len(Val)):
		raise exceptions.SyntaxError('Invalid Parameters')
	return dict(zip(Cmd, Val))

Models = {'Logistic Regression' : linear_model.LogisticRegression(),
		  'Decision Tree Classification' : tree.DecisionTreeClassifier(max_depth = 5),
		  'SVC' : svm.SVC(C = 0.65, probability = True),
		  'Random Forest Classification' : ensemble.RandomForestClassifier(),
		  'Gradient Boosting Classification' : ensemble.GradientBoostingClassifier(n_estimators = 50, max_depth = 1),
		  'Gaussian Naive Bayes' : naive_bayes.GaussianNB(),
		  'Bernoulli Naive Bayes' : naive_bayes.BernoulliNB(),
		  'KNN' : neighbors.KNeighborsClassifier(),
		  }
Features = {'B' : 'base.csv',
			'S' : 'noise.csv',
			'L' : 'lang.csv',
			'A' : 'asr.csv',
			}

def RunExp(StrModel:str, Param:str, FeaUsed:list, DataPath:str, Label:str, std:bool = False, N:int = 0):
	Data = np.genfromtxt(DataPath + Label, delimiter = ',', dtype = int)
	Data = Data[:, np.newaxis]

	for f in FeaUsed:
		T = (np.genfromtxt(DataPath + Features[f], delimiter = ',' , dtype = float))
		if len(T.shape) < 2:
			T = T[:, np.newaxis]
		Data = np.concatenate((Data, T), axis = 1)
	
	if N > 0:
		Data = Data[:N, :]

	Lbl = Data[:, 0]
	Fea = Data[:,1:]
	if std:
		scaler = preprocessing.StandardScaler()
		Fea = scaler.fit_transform(Fea)

	Model = base.clone(Models[StrModel])
	SetParam(Model, Param)

	Model.fit(Fea, Lbl)
	Pred = Model.predict_proba(Fea)[:, 1]
	st = metrics.roc_auc_score(Lbl, Pred)
		
	sv = cross_validation.cross_val_score(base.clone(Models[StrModel]), Fea, Lbl, metrics.make_scorer(metrics.roc_auc_score), cv = 5, n_jobs = 5)
	
	return st, np.mean(sv)

def main(argc:int, argv:list):
	DataPath = 'Data/ted/'
	StrOut = 'Result_Prob.csv'
	StrModel = 'all'
	Feature = []
	Label = 'acc.csv'
	Parameters = ''
	std = True
	n = 0

	try:
		Params, args = getopt.getopt(argv[1:], 'm:f:p:l:d:o:s:n:')
	except getopt.GetoptError:
		print('Invalid Param')
		sys.exit(2)

	for Cmd, Val in Params:
		if Cmd == '-m':
			StrModel = Val
		if Cmd == '-f':
			Feature.append(Val)
		if Cmd == '-p':
			Parameters = Val
		if Cmd == '-l':
			Label = Val
		if Cmd == '-d':
			DataPath = Val
		if Cmd == '-o':
			StrOut = Val
		if Cmd == '-s':
			std = bool(Val)
		if Cmd == '-n':
			n = int(Val)


	print('Parameters: ', Parameters)

	ST = []
	SV = []
	ModelUsed = []
	FeaUsed = []
	for m in Models:
		if (StrModel == 'all' or StrModel == m):
			ModelUsed.append(m)
	ModelUsed = sorted(ModelUsed)
	if len(Feature) > 0:
		FeaUsed.append(Feature);
	else:
		AllFeas = sorted(list(Features.keys()))
		for i in range(1, len(AllFeas) + 1):
			FeaUsed += itertools.combinations(AllFeas, i)

	for m in ModelUsed:
		ST.append([])
		SV.append([])
		for f in FeaUsed:
			print('Model: ', m)
			print('Features: ', f)
			st, sv = RunExp(m, Parameters, f, DataPath, Label, std, n)
			ST[-1].append(st)
			SV[-1].append(sv)
			print('Train: ', st, '\t', 'Valid: ', sv)

	with open(DataPath + StrOut, 'w') as fout:
		for f in FeaUsed:
			fout.write('Model\\Features,\"' + (''.join(f)) + '\"')
		fout.write('\n')
		for i in range(len(ModelUsed)):
			fout.write(ModelUsed[i])
			for j in range(len(FeaUsed)):
				fout.write(',' + str(round(SV[i][j], 3)))
			fout.write('\n')
		
		for f in FeaUsed:
			fout.write(',')
		fout.write('\n')
		
		for f in FeaUsed:
			fout.write('Model\\Features,\"' + (''.join(f)) + '\"')
		fout.write('\n')
		for i in range(len(ModelUsed)):
			fout.write(ModelUsed[i])
			for j in range(len(FeaUsed)):
				fout.write(',' + str(round(ST[i][j], 3)))
			fout.write('\n')

if __name__ == '__main__':
	main(len(sys.argv), sys.argv)