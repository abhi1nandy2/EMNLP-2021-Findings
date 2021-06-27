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
from utils import match_path_with_section, get_acc, add_qid, get_split, get_dataset, get_qna_list, get_tfidf_vectorizer, get_tfidf_vector, get_section_features, get_q_features, get_q_section_dict, evaluate
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

models_list = ['roberta-base', 'bert-base-uncased']

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type = str, default = "", help = "path of the directory to save model and tokenizer")
parser.add_argument("--corpus_path", type=str, required=True, help = "path of the corpus json file")
parser.add_argument("--q_folderpath", type=str, required=True, help="path of the folder containing train, validation, test splits of the questions and all the questions in separate files")
parser.add_argument("--model_name", default = "roberta-base", type=str, help="type of sequence classification model used among {}".format(str(models_list)))
parser.add_argument("--from_pretrained", action="store_true", help="was the model initialized from base pretraining chkpt")
parser.add_argument("--is_tpu", action="store_true", help="Is tpu available")
parser.add_argument("--is_fine_tune", action="store_true", help="is there any fine-tuning")
parser.add_argument("--checkpoint_path", default="", type=str, help="load the model of the checkpoint, if any")
parser.add_argument("--batch_size", default=16, type = int, help = "Batch Size")
args = parser.parse_args()

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

with open(args.corpus_path, 'r') as f:
	corpus_dict = json.load(f)			

qna_list, train_indices, valid_indices, test_indices = get_qna_list(args.q_folderpath)
tfidf_vec = get_tfidf_vectorizer(corpus_dict)
df_section_feats = get_section_features(corpus_dict, tfidf_vec)
df_q_feats = get_q_features(qna_list, tfidf_vec)
q_section_dict = get_q_section_dict(qna_list, corpus_dict)

#Actual stuff starts here. max_sim_mat is a matrix containing top K sections for each question retrieved using tfidf.

topK = 10
max_sim_mat = get_acc(df_q_feats, df_section_feats, corpus_dict, qna_list, topK=topK)
print(max_sim_mat.shape)

#stratified split - two classes - question has ground truth section in top 10 retrieved or not
# strat_ = []
# for idx in range(max_sim_mat.shape[0]):
# 	actual_sec_id = q_section_dict['q_{}'.format(idx)].replace("section_", "")
# 	actual_sec_id = int(actual_sec_id)
# 	if actual_sec_id in max_sim_mat[idx].tolist():
# 		strat_.append(1)
# 	else:
# 		strat_.append(0)
# print((np.asarray(strat_)==1).sum())

#


df_train= get_split(train_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)
df_valid= get_split(valid_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)
df_test= get_split(test_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)

if args.checkpoint_path != "":
    if args.from_pretrained:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base-tok")
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained("embert", max_len=512) #load tokenizer pretrained on our data
else:
	if args.model_name == 'bert-base-uncased':
		tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)
	else:
		tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base-tok")

# Tokenize all of the sentences and map the tokens to thier word IDs.

train_dataset = get_dataset(df_train, tokenizer)
val_dataset = get_dataset(df_valid, tokenizer)
prediction_dataset = get_dataset(df_test, tokenizer)

print(len(train_dataset), len(val_dataset), len(prediction_dataset))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = args.batch_size

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
			train_dataset,  # The training samples.
			sampler = RandomSampler(train_dataset), # Select batches randomly
			batch_size = batch_size, # Trains with this batch size.
			num_workers=8
		)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
			val_dataset, # The validation samples.
			sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
			batch_size = batch_size, # Evaluate with this batch size.
			num_workers=8
		)


if args.checkpoint_path == "":
	model = BertForSequenceClassification.from_pretrained(
		args.model_name, # Use the 12-layer BERT model, with an uncased vocab.
		num_labels = 2, # The number of output labels--2 for binary classification.
						# You can increase this for multi-class tasks.   
		output_attentions = False, # Whether the model returns attentions weights.
		output_hidden_states = False, # Whether the model returns all hidden-states.
	)
else:
	if "more_grad" in args.checkpoint_path:
		config = RobertaConfig(max_position_embeddings=514, vocab_size=50265, type_vocab_size=1, num_labels = 2, output_attentions = False, output_hidden_states = False)
		model = RobertaForSequenceClassification(config = config)
		chkpt_wts = torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin'))
		transfer_wts = OrderedDict()
		for key in chkpt_wts:
			if 'base_model.roberta' in key:
				transfer_wts[key.replace('base_model.roberta.', '')] = chkpt_wts[key]

		model.roberta.load_state_dict(transfer_wts)		
	else:
		model = RobertaForSequenceClassification.from_pretrained(
			args.checkpoint_path, # Use the 12-layer BERT model, with an uncased vocab.
			num_labels = 2, # The number of output labels--2 for binary classification.
							# You can increase this for multi-class tasks.   
			output_attentions = False, # Whether the model returns attentions weights.
			output_hidden_states = False, # Whether the model returns all hidden-states.
		)

model.to(device)
print(model.device)
params = list(model.named_parameters())

print('The model has {:} different named parameters.\n'.format(len(params)))

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
				  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
				  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
				)

epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
											num_warmup_steps = 0, # Default value in run_glue.py
											num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))

import random
import numpy as np
from transformers import Trainer, TrainingArguments

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

if args.is_fine_tune:
	# We'll store a number of quantities such as training and validation loss, 
	# validation accuracy, and timings.
	training_stats = []

	# Measure the total training time for the whole run.
	total_t0 = time.time()

	# For each epoch...
	for epoch_i in range(0, epochs):
		
		# ========================================
		#               Training
		# ========================================
		
		# Perform one full pass over the training set.

		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Training...')

		# Measure how long the training epoch takes.
		t0 = time.time()

		# Reset the total loss for this epoch.
		total_train_loss = 0

		# Put the model into training mode. Don't be mislead--the call to 
		# `train` just changes the *mode*, it doesn't *perform* the training.
		# `dropout` and `batchnorm` layers behave differently during training
		# vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
		model.train()

		# For each batch of training data...
		for step, batch in enumerate(train_dataloader):

			# Progress update every 40 batches.
			if step % 40 == 0 and not step == 0:
				# Calculate elapsed time in minutes.
				elapsed = format_time(time.time() - t0)
				
				# Report progress.
				print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

			# Unpack this training batch from our dataloader. 
			#
			# As we unpack the batch, we'll also copy each tensor to the GPU using the 
			# `to` method.
			#
			# `batch` contains three pytorch tensors:
			#   [0]: input ids 
			#   [1]: token_type_ids
			#   [2]: attention masks
			#   [3]: labels 
			b_input_ids = batch[0].to(device)
			b_token_type_ids = batch[1].to(device)
			b_input_mask = batch[2].to(device)
			b_labels = batch[3].to(device)

			# Always clear any previously calculated gradients before performing a
			# backward pass. PyTorch doesn't do this automatically because 
			# accumulating the gradients is "convenient while training RNNs". 
			# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
			model.zero_grad()        

			# Perform a forward pass (evaluate the model on this training batch).
			# The documentation for this `model` function is here: 
			# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
			# It returns different numbers of parameters depending on what arguments
			# arge given and what flags are set. For our useage here, it returns
			# the loss (because we provided labels) and the "logits"--the model
			# outputs prior to activation.
			loss, logits = model(b_input_ids, 
								 token_type_ids=b_token_type_ids, 
								 attention_mask=b_input_mask, 
								 labels=b_labels)

			# Accumulate the training loss over all of the batches so that we can
			# calculate the average loss at the end. `loss` is a Tensor containing a
			# single value; the `.item()` function just returns the Python value 
			# from the tensor.
			total_train_loss += loss.item()

			# Perform a backward pass to calculate the gradients.
			loss.backward()

			# Clip the norm of the gradients to 1.0.
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

			# Update parameters and take a step using the computed gradient.
			# The optimizer dictates the "update rule"--how the parameters are
			# modified based on their gradients, the learning rate, etc.
			optimizer.step()

			# Update the learning rate.
			scheduler.step()

		# Calculate the average loss over all of the batches.
		avg_train_loss = total_train_loss / len(train_dataloader)            
		
		# Measure how long this epoch took.
		training_time = format_time(time.time() - t0)

		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epoch took: {:}".format(training_time))
			
		# ========================================
		#               Validation
		# ========================================
		# After the completion of each training epoch, measure our performance on
		# our validation set.

		print("")
		print("Running Validation...")

		t0 = time.time()

		# Put the model in evaluation mode--the dropout layers behave differently
		# during evaluation.
		model.eval()

		# Tracking variables 
		total_eval_accuracy = 0
		total_eval_loss = 0
		nb_eval_steps = 0

		# Evaluate data for one epoch
		for batch in validation_dataloader:
			
			# Unpack this training batch from our dataloader. 
			#
			# As we unpack the batch, we'll also copy each tensor to the GPU using 
			# the `to` method.
			#
			# `batch` contains three pytorch tensors:
			#   [0]: input ids 
			#   [1]: token_type_ids
			#   [2]: attention masks
			#   [3]: labels 
			b_input_ids = batch[0].to(device)
			b_token_type_ids = batch[1].to(device)
			b_input_mask = batch[2].to(device)
			b_labels = batch[3].to(device)
			
			# Tell pytorch not to bother with constructing the compute graph during
			# the forward pass, since this is only needed for backprop (training).
			with torch.no_grad():        

				# Forward pass, calculate logit predictions.
				# token_type_ids is the same as the "segment ids", which 
				# differentiates sentence 1 and 2 in 2-sentence tasks.
				# The documentation for this `model` function is here: 
				# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
				# Get the "logits" output by the model. The "logits" are the output
				# values prior to applying an activation function like the softmax.
				(loss, logits) = model(b_input_ids, 
									   token_type_ids=b_token_type_ids, 
									   attention_mask=b_input_mask,
									   labels=b_labels)
				
			# Accumulate the validation loss.
			total_eval_loss += loss.item()

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			# Calculate the accuracy for this batch of test sentences, and
			# accumulate it over all batches.
			total_eval_accuracy += flat_accuracy(logits, label_ids)
			

		# Report the final accuracy for this validation run.
		avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
		print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

		# Calculate the average loss over all of the batches.
		avg_val_loss = total_eval_loss / len(validation_dataloader)
		
		# Measure how long the validation run took.
		validation_time = format_time(time.time() - t0)
		
		print("  Validation Loss: {0:.2f}".format(avg_val_loss))
		print("  Validation took: {:}".format(validation_time))

		# Record all statistics from this epoch.
		training_stats.append(
			{
				'epoch': epoch_i + 1,
				'Training Loss': avg_train_loss,
				'Valid. Loss': avg_val_loss,
				'Valid. Accur.': avg_val_accuracy,
				'Training Time': training_time,
				'Validation Time': validation_time
			}
		)

	print("")
	print("Training complete!")

	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

if args.save_dir != "":
	model.save_prtrained(args.save_dir)
	tokenizer.save_pretrained(args.save_dir)

prediction_sampler = SequentialSampler(prediction_dataset)
prediction_dataloader = DataLoader(prediction_dataset, sampler=prediction_sampler, batch_size=batch_size, num_workers=8)