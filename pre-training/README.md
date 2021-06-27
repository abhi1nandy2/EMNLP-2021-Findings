## Pre-training

### Files

1. `training_utils.py` - Function returning Optimizer having linearly decaying learning rates across neural net layers
2. `pretrain_manuals.py` - This contains the code for starting and/or resuming pre-training from any checkpoint. Presently using RoBERTa. To run - 

```
python pretrain-manuals.py --checkpoint-path=<CHECKPOINT FOLDER NAME>
```
(Can also check the other arguments in the file).
