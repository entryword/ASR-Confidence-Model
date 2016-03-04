from get_answer import get_anwser

def Label(Prefix:str, Fbin, Fdis):
	# edit distance
	# Fbin

	edit_dist,bin_dist = list(zip(*get_anwser('.',Prefix)))

	for d in edit_dist:
		Fdis.write(str(d)+'\n')

	for d in bin_dist:
		Fbin.write(str(int(d))+'\n')


	return 0
