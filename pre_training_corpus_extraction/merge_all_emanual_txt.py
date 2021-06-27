import sys, json
from tqdm import tqdm
import requests
import os
import sys
from textblob import TextBlob
import nltk
nltk.download('punkt')

def isEnglish(s):
	try:
		s.encode(encoding='utf-8').decode('ascii')
	except UnicodeDecodeError:
		return False
	else:
		return True

def get_num_toks(sent_list):
	num_toks = 0
	for sent in sent_list:
		num_toks += len(sent.split(" "))
	return num_toks

txtname_list = []
per_folder = 8300
total_num_sents = 0
total_num_toks = 0

with open("../data/full_raw_corpus.txt", "w") as f:
	for i in tqdm(range(1, 39)):
		dirname = "../data/manuals_dump"
		fname_list = os.listdir(dirname)
		start_f = per_folder*(i-1) + 1
		end_f = per_folder*i
		for j in tqdm(range(start_f, end_f + 1)):
			fname = "{}.txt".format(j)
			if fname not in fname_list: #invalid file
				continue
			if fname in txtname_list: #duplicate
				continue
			txtname_list.append(fname)
			with open(os.path.join(dirname, fname), 'r') as fr:
				for line in fr:
					line = line.strip()
					sents = [item.raw for item in TextBlob(line).sentences]
					sents = [item for item in sents if isEnglish(item)]
					total_num_sents+=len(sents)
					total_num_toks+=get_num_toks(sents)
					f.write(" ".join(sents))
					f.write("\n")
			f.write("\n") #line break between manuals.
		os.system("rm -r {}".format(zipname))
		os.system("rm -r {}".format(dirname))

