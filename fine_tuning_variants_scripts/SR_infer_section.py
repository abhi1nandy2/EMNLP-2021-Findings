import torch
import argparse
import json
import os, sys
from collections import OrderedDict
# import networkx as nx
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from transformers import BertForSequenceClassification, AdamW, BertConfig, RobertaConfig, RobertaForSequenceClassification
from transformers import RobertaTokenizerFast, BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils import match_path_with_section, get_acc, add_qid, get_split, get_dataset, get_qna_list, get_tfidf_vectorizer, get_tfidf_vector, get_section_features, get_q_features, get_q_section_dict, evaluate
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

parser = argparse.ArgumentParser()
parser.add_argument("--ir_chkpt", type = str, required = True, help = "path of the directory containing IR model and tokenizer")
parser.add_argument("--save_dir", type = str, required = True, help = "path where train, test and valid dictionaries would be saved")
parser.add_argument("--corpus_path", type=str, required=True, help = "path of the corpus json file")
parser.add_argument("--q_folderpath", type=str, required=True, help="path of the folder containing train, validation, test splits of the questions and all the questions in separate files")
parser.add_argument("--is_tpu", action="store_true", help="Is tpu available")

args = parser.parse_args()

with open(args.corpus_path, 'r') as f:
	corpus_dict = json.load(f)

if os.path.exists(args.save_dir) == False:
	os.mkdir(args.save_dir)

if args.is_tpu:
	import torch_xla.core.xla_model as xm
	device = xm.xla_device()

# If there's a GPU available...
elif torch.cuda.is_available():    

	# Tell PyTorch to use the GPU.    
	device = torch.device("cuda")

	print('There are %d GPU(s) available.' % torch.cuda.device_count())

	print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")

print(device)

qna_list, train_indices, valid_indices, test_indices = get_qna_list(args.q_folderpath)
tfidf_vec = get_tfidf_vectorizer(corpus_dict)
df_section_feats = get_section_features(corpus_dict, tfidf_vec)
df_q_feats = get_q_features(qna_list, tfidf_vec)
q_section_dict = get_q_section_dict(qna_list, corpus_dict)

topK = 10
max_sim_mat = get_acc(df_q_feats, df_section_feats, corpus_dict, qna_list, topK=topK)
print(max_sim_mat.shape)

df_train= get_split(train_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)
df_valid= get_split(valid_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)
df_test= get_split(test_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)

tokenizer = RobertaTokenizerFast.from_pretrained(args.ir_chkpt, max_len=512)

train_dataset = get_dataset(df_train, tokenizer)
val_dataset = get_dataset(df_valid, tokenizer)
test_dataset = get_dataset(df_test, tokenizer)

print(len(train_dataset), len(val_dataset), len(test_dataset))

batch_size = 16

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
			train_dataset,  # The training samples.
			sampler = RandomSampler(train_dataset), # Select batches randomly
			batch_size = batch_size # Trains with this batch size.
		)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
			val_dataset, # The validation samples.
			sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
			batch_size = batch_size # Evaluate with this batch size.
		)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

#load model

model = RobertaForSequenceClassification.from_pretrained(
	args.ir_chkpt, # Use the 12-layer BERT model, with an uncased vocab.
	num_labels = 2, # The number of output labels--2 for binary classification.
					# You can increase this for multi-class tasks.   
	output_attentions = False, # Whether the model returns attentions weights.
	output_hidden_states = False, # Whether the model returns all hidden-states.
)

train_list = evaluate(model, device, train_indices, train_dataloader, max_sim_mat, qna_list, q_section_dict, stage = 'train', get_relevant_section = True, num_rel = 5)
valid_list = evaluate(model, device, valid_indices, validation_dataloader, max_sim_mat, qna_list, q_section_dict, stage = 'valid', get_relevant_section = True, num_rel = 5)
test_list = evaluate(model, device, test_indices, test_dataloader, max_sim_mat, qna_list, q_section_dict, stage = 'test', get_relevant_section = True, num_rel = 5)

with open(os.path.join(args.save_dir, "train_Q_A.json"), 'w') as f:
	json.dump(train_list, f, indent = 2)

with open(os.path.join(args.save_dir, "valid_Q_A.json"), 'w') as f:
	json.dump(valid_list, f, indent = 2)

with open(os.path.join(args.save_dir, "test_Q_A.json"), 'w') as f:
	json.dump(test_list, f, indent = 2)
