import os, sys
# os.system("pip install transformers==2.11.0")
from datasets import load_dataset
import argparse
import numpy as np
import torch
import torch.nn as nn
import transformers
import datasets
import logging
from mtl_model import MultitaskModel
import pickle
from dataclasses import dataclass

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help = "data directory containing csvs")
parser.add_argument("--checkpoint_path", type=str, required = True, help= "path of pre-trained checkpoint")
# parser.add_argument("--trained_chkpt", type=str, required = True, help= "path of saved checkpoint")
parser.add_argument("--is_train", action="store_true", help="would we train?")
parser.add_argument("--is_tpu", action="store_true", help="Is tpu available")
args = parser.parse_args()

if os.path.exists('ir_dataset.pickle') == False:
	#ir
	ir_dataset = load_dataset('csv', data_files = {'train': os.path.join(args.data_dir, 'ir_train.csv'),
													'validation': os.path.join(args.data_dir, 'ir_valid.csv'),
													'test': os.path.join(args.data_dir, 'ir_test.csv')})
	print(ir_dataset)
	with open('ir_dataset.pickle', 'wb') as f:
	    # Pickle the 'data' dictionary using the highest protocol available.
	    pickle.dump(ir_dataset, f, pickle.HIGHEST_PROTOCOL)

else:
	with open('ir_dataset.pickle', 'rb') as f:
	    # The protocol version used is detected automatically, so we do not
	    # have to specify it.
	    ir_dataset = pickle.load(f)

if os.path.exists('rc_dataset.pickle') == False:
	#rc
	rc_dataset = load_dataset('csv', data_files = {'train': os.path.join(args.data_dir, 'rc_train.csv'),
													'validation': os.path.join(args.data_dir, 'rc_valid.csv'),
													'test': os.path.join(args.data_dir, 'rc_test.csv')})

	print(rc_dataset)
	with open('rc_dataset.pickle', 'wb') as f:
	    # Pickle the 'data' dictionary using the highest protocol available.
	    pickle.dump(rc_dataset, f, pickle.HIGHEST_PROTOCOL)

else:
	with open('rc_dataset.pickle', 'rb') as f:
	    # The protocol version used is detected automatically, so we do not
	    # have to specify it.
	    rc_dataset = pickle.load(f)

dataset_dict = {'ir': ir_dataset, 'rc': rc_dataset}

# raise ValueError("STOP")

if args.is_train:
	model_name = args.checkpoint_path
	multitask_model = MultitaskModel.create(
	    model_name=model_name,
	    model_type_dict={
	        "ir": transformers.AutoModelForSequenceClassification,
	        "rc": transformers.AutoModelForSequenceClassification,
	    },
	    model_config_dict={
	        "ir": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
	        "rc": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
	    },
	)
else:
	multitask_model = MultitaskModel.from_pretrained(args.checkpoint_path)

tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base-tok')

max_length = 512

def convert_to_ir_features(example_batch):
    inputs = list(zip(example_batch['sentence_1'], example_batch['sentence_2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_rc_features(example_batch):
    inputs = list(zip(example_batch['sentence_1'], example_batch['sentence_2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

convert_func_dict = {
    "ir": convert_to_ir_features,
    "rc": convert_to_rc_features
}

columns_dict = {
    "ir": ['input_ids', 'attention_mask', 'labels'],
    "rc": ['input_ids', 'attention_mask', 'labels']
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        features_dict[task_name][phase].set_format(
            type="torch", 
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))


import dataclasses
from torch.utils.data.dataloader import DataLoader
# from transformers.training_args import is_tpu_available
from transformers.trainer import get_tpu_sampler
from transformers import DataCollator
from transformers.data.data_collator import InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict

class NLPDataCollator:
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """
    def collate_batch(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
          # NLP data sets current works presents features as lists of dictionary
          # (one per example), so we  will adapt the collate_batch logic for that
          if "labels" in first and first["labels"] is not None:
              if first["labels"].dtype == torch.int64:
                  labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
              else:
                  labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
              batch = {"labels": labels}
          for k, v in first.items():
              if k != "labels" and v is not None and not isinstance(v, str):
                  batch[k] = torch.stack([f[k] for f in features])
          return batch
        else:
          # otherwise, revert to using the default collate_batch
          return DefaultDataCollator().collate_batch(features)

data_collator_ = NLPDataCollator()

class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])    

class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if args.is_tpu:
            train_sampler = get_tpu_sampler(train_dataset)
        else:
            train_sampler = (
                RandomSampler(train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(train_dataset)
            )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=data_collator_.collate_batch,
            ),
        )

        if args.is_tpu:
            data_loader = pl.ParallelLoader(
                data_loader, [self.args.device]
            ).per_device_loader(self.args.device)
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })

if args.is_train:
	train_dataset = {
	    task_name: dataset["train"] 
	    for task_name, dataset in features_dict.items()
	}
	trainer = MultitaskTrainer(
	    model=multitask_model,
	    args=transformers.TrainingArguments(
	        output_dir="./multitask_model",
	        overwrite_output_dir=True,
	        learning_rate=1e-5,
	        do_train=True,
	        num_train_epochs=4,
	        # Adjust batch size if this doesn't fit on the Colab GPU
	        per_device_train_batch_size=8,  
	        save_steps=16,
	        save_total_limit=1
	    ),
	    data_collator=data_collator_,
	    train_dataset=train_dataset,
	)
	trainer.train()

	# os.system("pip install transformers")

else:
	preds_dict = {}
	for task_name in ["ir", "rc"]:
	    eval_dataloader = DataLoaderWithTaskname(
	        task_name,
	        trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["test"])
	    )
	    print(eval_dataloader.data_loader.collate_fn)
	    preds_dict[task_name] = trainer._prediction_loop(
	        eval_dataloader, 
	        description=f"Validation: {task_name}",
	    )
	    print(preds_dict)
	    print(preds_dict.keys())