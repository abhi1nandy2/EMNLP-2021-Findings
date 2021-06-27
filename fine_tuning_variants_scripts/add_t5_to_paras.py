import os
import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import textwrap
from tqdm import tqdm
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", type = str, required= True, help = "Directory containing T5 base model")
parser.add_argument("--corpus_file", type = str, required = True, help = ".json File containing sectionwise emanual corpus")

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

prev_wdir = os.getcwd()
os.chdir(args.model_dir)
os.system("wget https://storage.googleapis.com/doctttttquery_git/t5-base.zip")
os.system("unzip t5-base")
os.system("rm -r t5-base.zip")
os.chdir(prev_wdir)

class docT5Query():
	def __init__(self, model_weight, from_tf, device):
		self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
		self.config = T5Config.from_pretrained('t5-base')
		self.device = device
		self.model = T5ForConditionalGeneration.from_pretrained(model_weight, from_tf=from_tf, config=self.config)
		self.model.to(self.device)

	def predict(self, paragraph, max_length=64, num_query=5):
		input_ids = self.tokenizer.encode(paragraph, return_tensors='pt').to(self.device)
		outputs = self.model.generate(input_ids=input_ids, max_length=max_length, do_sample=True, top_k=10, num_return_sequences=num_query)
		queries = []
		for i in range(num_query):
			queries.append(self.tokenizer.decode(outputs[i], skip_special_tokens=True))
		return queries

#getting the model wt. name

model_files = os.listdir(args.model_dir)

for file in model_files:
	if ".meta" in file:
		model_weight = os.path.join(args.model_dir, file).replace(".meta", "")
		break

query_generator = docT5Query(model_weight, True, device)

with open(args.corpus_file, 'r') as f:
	corpus_dict = json.load(f)

for sec_id in tqdm(corpus_dict):
	p = " ".join(corpus_dict[sec_id]['text'])
	queries = query_generator.predict(p)
	p = p + " " + ' ? '.join(queries).strip().lower()
	corpus_dict[sec_id]["t5_para"] = p

with open(args.corpus_file, 'w') as f:
	json.dump(corpus_dict, f, indent = 2)