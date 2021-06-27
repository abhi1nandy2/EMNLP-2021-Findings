import os, sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm
import json
import rouge
import string
from collections import OrderedDict
from transformers import RobertaTokenizerFast

def float_to_str(num):
    str_ = "{:.3f}".format(float(num))
    return str_

def remove_punct(s):
    s = s.replace(" :", ":").replace(" .", ".").replace(" .", ".").replace("’", "").replace("'", "").replace("Wi‑Fi", "WiFi").split()
    s = [i for i in s if i != " "]
    s = ' '.join(s)
    s = ". ".join(s.split("."))
    s = ", ".join(s.split(","))
    # _RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")
    # _RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")
    # s = s.replace("Wi‑Fi", "WiFi") #domain specific
    # s = s.replace(".", ". ")
    temp_sent = s.replace(" :", ":").replace(" ,", ",").replace(" .", ".").replace(" .", ".").replace("’", "").replace("'", "").replace("Wi‑Fi", "WiFi").split()
    temp_sent = [i for i in temp_sent if i != " "] #remove extra spaces
    # s = _RE_COMBINE_WHITESPACE.sub(" ", s)
    # s = _RE_STRIP_WHITESPACE.sub("", s)
    s = ' '.join(temp_sent)
    
    s = s.translate(str.maketrans('', '', string.punctuation)) #remove all punct.
    return s

def match_path_with_section(q_section_path, corpus_dict):
    q_section_path = q_section_path.replace("‑", "").replace("?", "")
    q_section_path = q_section_path.split(">")
    q_section_path = [item.strip() for item in q_section_path]
    q_section_path = [item for item in q_section_path if item!=""]
    section_name = q_section_path[-1]
    for iter_sec in corpus_dict:
        iter_sec_content = corpus_dict[iter_sec]
        if section_name == iter_sec_content['title']:
            iter_sec_path = iter_sec_content['section_hierarchy'].split(">")
            iter_sec_path = [corpus_dict[item]['title'] for item in iter_sec_path]
            if iter_sec_path[0:3] == q_section_path[0:3]:
                return iter_sec_content['id']
    return None #if nothing matches

def get_acc(df_q_feats, df_node_feats, corpus_dict, qna_list, topK = 5):
    correct = 0
    total = df_q_feats.shape[0]

    q_arr = np.asarray(df_q_feats)
    node_arr = np.asarray(df_node_feats[:][:-1])

    cos_sim_mat = cosine_similarity(q_arr, node_arr)
    # print(cos_sim_mat.shape)

    # max_sim = np.argmax(cos_sim_mat, axis = 1)
    max_sim = np.argsort(cos_sim_mat, axis = 1)[:,-topK:][:, ::-1]
    # print(max_sim.shape)

    for idx in range(total):
        pred_sections = ["section_{}".format(max_sim[idx][k]) for k in range(topK)]
        actual_section = match_path_with_section(qna_list[idx]['Section Hierarchy'], corpus_dict)
        if actual_section in pred_sections:
            # print(idx)
            correct+=1
    print(correct, correct/total)
    return max_sim

def add_qid(q_dict, indices):
    q_list = q_dict['All_Questions']
    for idx in range(len(q_list)):
        q_list[idx]['qid'] = "q_" + str(indices[idx]) #len of q_dict and indices are the same

    q_dict['All_Questions'] = q_list
    return q_dict

def get_split(indices, q_section_dict, max_sim_mat, qna_list, corpus_dict):
    sent1 = []
    sent2 = []
    labels = []
    for idx in indices:

        actual_sec_id = q_section_dict["q_" + str(idx)].replace("section_", "")
        # idx = int(idx.replace("q_", ""))
        actual_sec_id = int(actual_sec_id)
        for sec_id in max_sim_mat[idx].tolist():
            sent1.append(qna_list[idx]['Question'])
            sent2.append(corpus_dict["section_{}".format(sec_id)]['t5_para'])
            if sec_id == actual_sec_id:
                labels.append(1)
            else:
                labels.append(0)
    df = pd.DataFrame(columns = ['sentence_1', 'sentence_2', 'label'])
    df['sentence_1'] = sent1
    df['sentence_2'] = sent2
    df['label'] = labels
    print(df.head())
    print(df.shape)
    print((np.asarray(labels)==1).sum())
    return df

def get_split_for_rc(stage, relation_df, stage_q_dict, corpus_dict):
    sent1 = []
    sent2 = []
    labels = []
    # qids = []
    relation_df.reset_index(drop = True, inplace = True)
    for i in range(relation_df.shape[0])    :
        labels.append(relation_df['2'][i])
        temp = relation_df['1'][i]
        temp = temp.split("|")
        # if temp[0] not in qids:
        #     qids.append(temp[0])  #store uniqs
        # print(stage_q_dict)
        sent1.append(stage_q_dict[int(temp[0].replace("{}_Q".format(stage), ""))]["QUESTION_TEXT"])
        sent2.append(corpus_dict[temp[1]]['text'][int(temp[2])])

    df = pd.DataFrame(columns = ['sentence_1', 'sentence_2', 'label'])
    df['sentence_1'] = sent1
    df['sentence_2'] = sent2
    df['label'] = labels
    print(df.head())
    print(df.shape)
    return df

def get_split_for_rc_tokwise(stage, relation_df, stage_q_dict, corpus_dict):
    sent1 = []
    sent2 = []
    labels = []
    qid_q = OrderedDict()
    qid_labels = OrderedDict()
    qid_section_dict = OrderedDict()
    # qids = []
    relation_df.reset_index(drop = True, inplace = True)
    for i in range(relation_df.shape[0])    :
        # labels.append(relation_df['2'][i])
        temp = relation_df['1'][i]
        temp = temp.split("|")
        # if temp[0] not in qids:
        #     qids.append(temp[0])  #store uniqs
        # print(stage_q_dict)
        qid = temp[0]
        q = stage_q_dict[int(temp[0].replace("{}_Q".format(stage), ""))]["QUESTION_TEXT"]
        ans_sent = corpus_dict[temp[1]]['text'][int(temp[2])]
        ans_sent = remove_punct(ans_sent)
        if qid not in qid_section_dict:
            gt_ans = stage_q_dict[int(qid.replace("{}_Q".format(stage), ""))]["ANSWER"]
            qid_q[qid] = q
            qid_section_dict[qid] = [ans_sent]
            if ans_sent in remove_punct(gt_ans):
                qid_labels[qid] = [1]
            else:
                qid_labels[qid] = [0]
        else:
            qid_section_dict[qid].append(ans_sent)
            if ans_sent in remove_punct(gt_ans):
                qid_labels[qid].append(1)
            else:
                qid_labels[qid].append(0)
    for qid in qid_section_dict:
        sent1.append(qid_q[qid])
        sent2.append(qid_section_dict[qid])
        labels.append(qid_labels[qid])

    df = pd.DataFrame(columns = ['sentence_1', 'sentence_2', 'label'])
    df['sentence_1'] = sent1
    df['sentence_2'] = sent2
    df['label'] = labels
    print(df.head())
    print(df.shape)
    return df

def get_dataset(df, tokenizer):
    sentences_1 = df.sentence_1.values
    sentences_2 = df.sentence_2.values
    labels = df.label.values
    input_ids = []
    attention_masks = []
    token_type_ids = []

    # For every sentence...
    for sent_idx in tqdm(range(len(sentences_1))):
        # inp = sentences_1[sent_idx] + '[SEP]'+ sentences_2[sent_idx]
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer(
                            sentences_1[sent_idx],                      # Input to encode.
                            sentences_2[sent_idx],
                            add_special_tokens = True, # To Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_token_type_ids = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        token_type_ids.append(encoded_dict['token_type_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences_1[0], sentences_2[0])
    print('Token IDs:', input_ids[0])

    print(input_ids.shape, token_type_ids.shape, attention_masks.shape, labels.shape)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
    return dataset

def get_dataset_tokwise(df, tokenizer):
    sentences_1 = df.sentence_1.values
    sentences_2 = df.sentence_2.values
    init_labels = df.label.values
    labels = []
    input_ids = []
    attention_masks = []
    token_type_ids = []

    # For every sentence...
    for sent_idx in tqdm(range(len(sentences_1))):
        temp_labels = []
        # inp = sentences_1[sent_idx] + '[SEP]'+ sentences_2[sent_idx]
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        sentences = [sentences_1[sent_idx]] + sentences_2[sent_idx]
        compound = "<s> " + "</s></s> ".join(sentences) + "</s>"
        encoded_dict = tokenizer(
                            compound,
                            add_special_tokens = False, # To Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_token_type_ids = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        inp_ = encoded_dict['input_ids'].view(-1).tolist()
        # print(inp_)
        count_sep = 0
        count_tok = 0
        for idx, item in enumerate(inp_):
            count_tok += 1
            if item == 2 and inp_[idx - 1] == 2 and count_sep == 0:
                temp_labels.extend([2]*count_tok)
                count_tok = 0
                count_sep += 1
            elif (item == 2 and inp_[idx - 1] == 2) or (item == 2 and inp_[idx + 1] == 1):
                temp_labels.extend([init_labels[sent_idx][count_sep - 1]]*count_tok)
                count_tok = 0
                count_sep += 1
        temp_labels.extend([3]*count_tok)
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        token_type_ids.append(encoded_dict['token_type_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        labels.append(temp_labels)

    # Convert the lists into tensors.
    print(encoded_dict['input_ids'].shape)
    print(np.array(labels).shape)
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    # print('Original: ', sentences_1[0], sentences_2[0])
    print('Token IDs:', input_ids[0])

    print(input_ids.shape, token_type_ids.shape, attention_masks.shape, labels.shape)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
    return dataset

def get_qna_list(q_folderpath):
    start = 0

    with open(os.path.join(q_folderpath, "train_annotation.json")) as f:
        train_q_dict = json.load(f)
        len_train = len(train_q_dict['All_Questions'])
        train_indices = [i for i in range(start, start + len_train)]
        train_q_dict = add_qid(train_q_dict, train_indices)

        start += len_train

    with open(os.path.join(q_folderpath, "valid_annotation.json")) as f:
        valid_q_dict = json.load(f)
        len_valid = len(valid_q_dict['All_Questions'])
        valid_indices = [i for i in range(start, start + len_valid)]
        valid_q_dict = add_qid(valid_q_dict, valid_indices)

        start += len_valid

    with open(os.path.join(q_folderpath, "test_annotation.json")) as f:
        test_q_dict = json.load(f)
        len_test = len(test_q_dict['All_Questions'])
        test_indices = [i for i in range(start, start + len_test)]
        test_q_dict = add_qid(test_q_dict, test_indices)

    qna_list = []
    qna_list.extend(train_q_dict['All_Questions'])
    qna_list.extend(valid_q_dict['All_Questions'])
    qna_list.extend(test_q_dict['All_Questions'])
    return qna_list, train_indices, valid_indices, test_indices

def get_tfidf_vectorizer(corpus_dict):
    all_docs = []
    for section in corpus_dict:
        all_docs.append(corpus_dict[section]['t5_para'])
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_df=.75, min_df=1, stop_words='english', use_idf=True, norm=None, sublinear_tf=True)
    vectorizer.fit(all_docs)
    return vectorizer

def get_tfidf_vector(str_, tfidf_vec):
    return tfidf_vec.transform([str_]).toarray().reshape((-1,)).tolist()

def get_section_features(corpus_dict, tfidf_vec=None):
    vec_list = []
    sec_list = []
    for section in corpus_dict:
        sec_list.append(section)
        temp = corpus_dict[section]
        temp = temp['t5_para']
        temp = get_tfidf_vector(temp, tfidf_vec)
        vec_list.append(temp)
        # print(section)
    vecs = np.asarray(vec_list)
    print(vecs.shape)
    col_names = ["f{}".format(i) for i in range(1, vecs.shape[1] + 1)]
    index = sec_list
    df = pd.DataFrame(data = vecs, index = index, columns = col_names)
    return df

def get_q_features(qna_list, tfidf_vec=None):
    vec_list = []
    q_list = [] 
    for i in range(len(qna_list)):
        q_list.append("q_{}".format(i))
        temp = qna_list[i]['Question']
        temp = get_tfidf_vector(temp, tfidf_vec)
        vec_list.append(temp)
        # print(i)
    vecs = np.asarray(vec_list)
    col_names = ["f{}".format(i) for i in range(1, vecs.shape[1] + 1)]
    index = q_list
    df = pd.DataFrame(data = vecs, index = index, columns = col_names)
    return df

def get_q_section_dict(qna_list, corpus_dict):
    q_section_dict = {}

    for idx, item in tqdm(enumerate(qna_list)):
        section = match_path_with_section(item['Section Hierarchy'], corpus_dict)
        if (section == None):
            print("Wrong - {}".format(item['qid']))
            print(qna_list[idx]['Section Hierarchy'])
        else:
            q_section_dict[item['qid']] = section

    return q_section_dict