import os
import buildVector as bv
from get_answer import prune_asr_result
	
def computeAsr(fileName):
	#######
	global d
	d = bv.makeDictionary()
	dim=200

	######
	myfile = open(fileName)
	line = myfile.readline()
	plain_text = prune_asr_result(line)
	plain_text = plain_text.lower()
	word = plain_text.split(' ')

	ans = [0] * dim

	i=0
	while (i<len(word)):
		vt = d.get(word[i],[0]*dim )
		tmp = [ vt + ans for vt, ans in zip(vt, ans)]
		ans = tmp
		#dot = sum(ans*ans for ans, ans in zip(ans,ans))
		i+=1

	return ans

def computeAns(fileName):
	#######
	global d
	d = bv.makeDictionary()
	dim=200
	######
	myfile = open(fileName)
	line = myfile.readline()
	line = myfile.readline() ##jump first line about time
	plain_text = prune_asr_result(line)
	plain_text = plain_text.lower()
	word = plain_text.split(' ')

	ans = [0] * dim
	i=0
	while (i<len(word)):
		vt = d.get(word[i],[0]*dim )
		tmp = [ vt + ans for vt, ans in zip(vt, ans)]
		ans = tmp
		#dot = sum(ans*ans for ans, ans in zip(ans,ans))
		i+=1

	return ans

def vectorFeature(prefix, Fout):
    i = 1
    while (True):
        asr = prefix + str(i) + '.asr'
        ans = prefix + str(i) + '.ans'
        if not (os.path.isfile(ans) and os.path.isfile(asr)):
            break;
        vecAsr = computeAsr(asr)
        print(asr)
        vecAns = computeAns(ans)
        dot = sum(vecAsr*vecAns for vecAsr, vecAns in zip(vecAsr,vecAns))

        # you can assume anything is in the working directory
        features = dot
        Fout.write(str(features) + '\n')
        i += 1
