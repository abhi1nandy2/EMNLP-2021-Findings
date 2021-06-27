import sys, os
import numpy as np
import torch
import torch.nn as nn
import transformers
import datasets
import logging
from MTL_utils import *
import argparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", type=str, required=True, help = "path of the corpus json file")
parser.add_argument("--ir_q_folderpath", type=str, required=True, help="path of the folder containing train, validation, test splits of the questions and all the questions in separate files")
parser.add_argument("--rc_q_folderpath", type=str, required=True, help="path of the folder containing train, validation, test splits of the questions and all the questions in separate files")
parser.add_argument("--input_dir", type=str, required=True, help = "where to get input jsons")
parser.add_argument("--save_dir", type=str, required=True, help = "where to save dataset dataframe csvs")
args = parser.parse_args()

with open(args.corpus_path, 'r') as f:
	corpus_dict = json.load(f)			

qna_list, train_indices, valid_indices, test_indices = get_qna_list(args.ir_q_folderpath)
tfidf_vec = get_tfidf_vectorizer(corpus_dict)
df_section_feats = get_section_features(corpus_dict, tfidf_vec)
df_q_feats = get_q_features(qna_list, tfidf_vec)
q_section_dict = get_q_section_dict(qna_list, corpus_dict)

#Actual stuff starts here. max_sim_mat is a matrix containing top K sections for each question retrieved using tfidf.

topK = 10
max_sim_mat = get_acc(df_q_feats, df_section_feats, corpus_dict, qna_list, topK=topK)
print(max_sim_mat.shape)

df_train= get_split(train_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)
df_valid= get_split(valid_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)
df_test= get_split(test_indices, q_section_dict, max_sim_mat, qna_list, corpus_dict)

os.system("mkdir -p {}".format(args.save_dir))
df_train.to_csv("{}/ir_train.csv".format(args.save_dir), index_label = "idx") #also add index column
df_valid.to_csv("{}/ir_valid.csv".format(args.save_dir), index_label = "idx")
df_test.to_csv("{}/ir_test.csv".format(args.save_dir), index_label = "idx")

#now for rc

#have to add an arg here

mz_dir = args.input_dir

df_list = []
for stage in ['train', 'valid', 'test']:
	relation_file = os.path.join(mz_dir, stage, "relation_df.csv")
	relation_df = pd.read_csv(relation_file)
	with open(os.path.join(args.rc_q_folderpath, stage + "_Q_A.json"), 'r') as f:
		stage_q_dict = json.load(f)
	df = get_split_for_rc(stage, relation_df, stage_q_dict, corpus_dict)
	df_list.append(df)

df_train = df_list[0]
df_valid = df_list[1]
df_test = df_list[2]

os.system("mkdir -p {}".format(args.save_dir))
df_train.to_csv("{}/rc_train.csv".format(args.save_dir), index_label = "idx") #also add index column
df_valid.to_csv("{}/rc_valid.csv".format(args.save_dir), index_label = "idx")
df_test.to_csv("{}/rc_test.csv".format(args.save_dir), index_label = "idx")

# python mtl_data.py --ir_q_folderpath ../data/total_data_split --rc_q_folderpath ../data/ s10_techqa_from_pretrained_bsize_16 --corpus_path ../extract_Samsung_manuals_utils/data_extraction/temp_corpus.json --save_dir ../data/s10_mtl_from_pretrained_bsize_16