# Question Answering over Electronic Devices: A New Benchmark Dataset and a Multi-Task Learning based QA Framework

This repo has the code for the paper **"Question Answering over Electronic Devices: A New Benchmark Dataset and a Multi-Task Learning based QA Framework"** accepted at **EMNLP 2021 Findings**. The blog on this paper can be found [here](https://medium.com/@nandyabhilash/question-answering-over-electronic-devices-a-new-benchmark-dataset-and-a-multi-task-learning-based-5ac5661dc858).

## Required dependencies -

Please run `pip install -r requirements.txt` (`python3` required)

### E-Manual pre-training corpus

Go to [this link](https://drive.google.com/drive/folders/1-gX1DlmVodP6OVRJC3WBRZoGgxPuJvvt?usp=sharing). A RoBERTa BASE Model pre-trained on the corpus can be found [here](https://huggingface.co/AnonymousSub/EManuals_Roberta).

### Codes

- Annotated Data and Amazon User Forum Data Samples are present in [`data`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/data) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/data/README.md))
- Data Analysis is done in [`data_analysis`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/data_analysis) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/data_analysis/README.md))
- Corpus extraction code is present in [`pre_training_corpus_extraction`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/pre_training_corpus_extraction) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/pre_training_corpus_extraction/README.md))
- E-Manual Data Extraction code is present in [`EManual_data_extraction`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/EManual_data_extraction) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/EManual_data_extraction/README.md))
- Code on pre-training is given in [`pre-training`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/pre-training) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/pre-training/README.md))
- Code on unsupervised IR method and fine-tuning variants is given in [`fine_tuning_variants_scripts`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/fine_tuning_variants_scripts) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/fine_tuning_variants_scripts/README.md))
- Code on multi-task learning is given in [`MTL_scripts`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/MTL_scripts) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/MTL_scripts/README.md))
- Code on funtions for evaluation of MTL and fine-tuning variants is given in [`evaluation`](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/evaluation) (See [README](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/tree/main/evaluation/README.md))
  - For ROUGE-L Precision, Recall and F1-Score: https://pypi.org/project/py-rouge/
  - For S+WMS: https://github.com/eaclark07/sms

### Baselines

1. **Dense Passage Retrieval(DPR)** - Used HuggingFace implementation (https://huggingface.co/transformers/model_doc/dpr.html)
2. **Technical Answer Prediction (TAP)** - took the help of code in https://github.com/IBM/techqa
3. **MultiSpan** - took the help of code in https://github.com/eladsegal/tag-based-multi-span-extraction
