import os
import argparse
from training_utils import return_layerwise_decay_optim


parser = argparse.ArgumentParser()
parser.add_argument("--more_grad", action="store_true", help="applying a model similar to GoogleNet")
parser.add_argument("--from_latest", action="store_true", help="train from latest checkpoint")
parser.add_argument("--checkpoint_path", default="", type=str, help="load the model of the checkpoint (if training from a checkpoint), overrides `--from_latest`")
parser.add_argument("--from_pretrained", action="store_true", help="starting from a pretrained LM")
parser.add_argument("--per_dev_batch_size", default = 16, type = int, help="per device train batch size")
parser.add_argument("--layerwise_lr_decay", action = "store_true", help = "whether to have a layerwise lr decay - https://github.com/aws-health-ai/multi_domain_lm#learning-rate-control")
parser.add_argument("--ewc", action = "store_true", help = "Are we using Elastic Weight Consolidation")
parser.add_argument("--logging_dir", type=str, required = True, help = "directory for logging training runs")

args = parser.parse_args()

if args.from_pretrained:
    if args.more_grad:
        output_dir = "embert_model_from_pretrained_more_grad"
    elif args.layerwise_lr_decay:
        output_dir = "embert_model_from_pretrained_layerwise_lr_decay"
    elif args.ewc:
        output_dir = "embert_model_from_pretrained_ewc"
    else:
        output_dir = "embert_model_from_pretrained"
else:
    output_dir = "embert_model"

if args.checkpoint_path != "":
    checkpoint_path = args.checkpoint_path
elif args.from_latest:
    files = os.listdir(output_dir)
    chk_nums = [int(file.replace("checkpoint-", "")) for file in files]
    checkpoint_path = os.path.join(output_dir, "checkpoint-" + str(max(chk_nums)))
else:
    checkpoint_path = ""

import glob

import torch
import transformers
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import multiprocessing as mp #to see the number of cores

from copy import deepcopy

from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from dataclasses import dataclass
from copy import deepcopy

class linear(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        #print(features)
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

class more_grad_roberta(nn.Module):
    def __init__(self, config, layer_nums: List = [6, 12]):
        super(more_grad_roberta, self).__init__()
        #super().__init__(transformers.PretrainedConfig())
        self.layer_nums = layer_nums
        if os.path.exists("roberta-base_from_pretrained"):
            self.base_model = RobertaForMaskedLM.from_pretrained("roberta-base_from_pretrained", output_hidden_states=True)
        else:
            self.base_model = RobertaForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True)
        
        # self.linears = [nn.Linear(config.hidden_size, config.vocab_size, bias = False) for i in range(len(self.layer_nums))]
        self.head1 = linear(config)
        self.head2 = linear(config)
        self.head3 = linear(config)
        #for i in range(len(self.linears)):
            #self.linears[i].bias = nn.Parameter(torch.zeros(config.vocab_size))
    def forward(self, input_ids, **kwargs):
        #print(x)
        hidden_states = self.base_model(input_ids)
        out_tuple = ()
        # for i in range(len(self.layer_nums)):
        #     layer_num  = self.layer_nums[i]
        hidden = hidden_states[1][self.layer_nums[0]]
        out1 = self.head1(hidden)
        hidden = hidden_states[1][self.layer_nums[1]]
        out2 = self.head2(hidden)
        # hidden = hidden_states[1][self.layer_nums[2]]
        # out3 = self.head3(hidden)
        out_tuple = (out1,out2)#,out3)
        return out_tuple

@dataclass
class custom_data_collator(DataCollatorForLanguageModeling):
    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Any]:
        #raise ValueError(str(examples))
        #print(examples[0].keys())
        if isinstance(examples[0], (dict, BatchEncoding)):
            # print('hello')
            # print(examples[0])
            # print('hello')
            #raise ValueError("keys ", str(examples[0].keys()))
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": (labels, labels)}
        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": batch, "labels": (labels, labels)}


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, trainer: Trainer, model: nn.Module, dataset: list):

        self.trainer = trainer
        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for i in range(len(self.dataset['input_ids'])):
            input = {k:self.dataset[k][i].view(1, -1) for k in self.dataset}
            self.model.zero_grad()
            loss = self.trainer.compute_loss(self.model, input)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        trainer = Trainer(model)
        #print(inputs['input_ids'].shape)
        ewc = EWC(trainer, model, inputs)
        return trainer.compute_loss(model, inputs) + ewc.penalty()

if "shards" not in os.listdir("./"):

#     make shards - comment if already done
    os.system("mkdir -p ./shards")
    os.system("split -a 4 -l 256000 -d full_raw_corpus.txt ./shards/shard_")
    os.system("rm full_raw_corpus.txt")

#Save Tokenizer - commented if already done
# paths = ["full_raw_corpus.txt"]

#keep the import however!
from tokenizers import ByteLevelBPETokenizer

# tokenizer = ByteLevelBPETokenizer()

# tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
#     "<s>",
#     "<pad>",
#     "</s>",
#     "<unk>",
#     "<mask>",
# ])
# os.system("mkdir -p embert")
# tokenizer.save_model("embert")

#reordering list of files in ascending order
files = glob.glob('shards/*')
ord_files = []
for i in range(len(files)):
    str_i = str(i)
    str_i = (4 - len(str_i))*'0' + str_i
    fpath = "shards/shard_" + str_i
    ord_files.append(fpath)

print(ord_files)

#Making dataset and data collator

from datasets import load_dataset
dataset = load_dataset("text.py", data_files=ord_files, split='train', cache_dir="datasets_cache")
print(dataset)

if args.from_pretrained:
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base-tok")
else:
    tokenizer = RobertaTokenizerFast.from_pretrained("embert", max_len=512) #load tokenizer pretrained on our data

def encode(examples):
  return tokenizer(examples['text'], truncation=True, padding='max_length')

dataset = dataset.map(encode, batched=True, num_proc = 8, cache_file_name="map_cache.arrow")

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask']) #to convert data from .pyarrow file into torch compatible format

#only doing mlm training
config = RobertaConfig(
    vocab_size=52000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1, #type_vocab_size is the vocab size for 'token_type_ids' (sentence embedding)
)
# print(config)

if args.more_grad:
    if checkpoint_path == "":
        config = RobertaConfig()
        model = more_grad_roberta(config)     
    else:
        model = more_grad_roberta.from_pretrained(checkpoint_path)
        #state_dict = torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin'))
        #model.load_state_dict(state_dict)

else:
    if checkpoint_path=="":
        if args.from_pretrained:
            if os.path.exists("roberta-base_from_pretrained"):
                model = RobertaForMaskedLM.from_pretrained("roberta-base_from_pretrained")          
            else:
                model = RobertaForMaskedLM.from_pretrained("roberta-base")
        else:
            model = RobertaForMaskedLM(config=config)
        
    else:
        model = RobertaForMaskedLM.from_pretrained(checkpoint_path)

if args.more_grad:
    data_collator = custom_data_collator(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )    
else:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

training_args = TrainingArguments(
    output_dir=output_dir,
    #overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=args.per_dev_batch_size,
#     tpu_num_cores=8,
#     debug=True,
    dataloader_num_workers = 16,
#     logging_steps=20,
    logging_dir=args.logging_dir,
    save_steps=1000,
    save_total_limit=10,
)

if args.layerwise_lr_decay:
    opt = return_layerwise_decay_optim(model)
#elif args.layerwise_lr_decay and (args.checkpoint_path!=""):
#    raise ValueError("Functionality under construction")
else:
    opt = None

if args.ewc:
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        optimizers=(opt,None)
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        optimizers=(opt, None)
    )

print(training_args.device)
print(training_args.train_batch_size)

if checkpoint_path=="":
    trainer.train()
else:
    trainer.train(checkpoint_path)
