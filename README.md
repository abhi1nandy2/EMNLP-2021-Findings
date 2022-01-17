# Question Answering over Electronic Devices: A New Benchmark Dataset and a Multi-Task Learning based QA Framework

This repo has the code for the [paper](https://arxiv.org/abs/2109.05897) **"Question Answering over Electronic Devices: A New Benchmark Dataset and a Multi-Task Learning based QA Framework"** accepted at **EMNLP 2021 Findings**. The blog on this paper can be found [here](https://medium.com/@nandyabhilash/question-answering-over-electronic-devices-a-new-benchmark-dataset-and-a-multi-task-learning-based-5ac5661dc858), the poster [here](https://github.com/abhi1nandy2/EMNLP-2021-Findings/blob/main/poster.pdf), and a corresponding presentation [here](https://docs.google.com/presentation/d/1k2qj6cmh3C1ZXmlYzscPlM3sRubD7kzqEDP2npvkK6c/edit?usp=sharing).

## Required dependencies -

Please run `pip install -r requirements.txt` (`python3` required)

### E-Manual pre-training corpus

Go to [this link](https://drive.google.com/drive/folders/1-gX1DlmVodP6OVRJC3WBRZoGgxPuJvvt?usp=sharing). A RoBERTa BASE Model pre-trained on the corpus can be found [here](https://huggingface.co/abhi1nandy2/EManuals_RoBERTa), and a BERT BASE UNCASED Model pre-trained on the same [here](https://huggingface.co/abhi1nandy2/EManuals_BERT).

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

## Citation

Please cite the work if you would like to use it.

```
@inproceedings{nandy-etal-2021-question-answering,
    title = "Question Answering over Electronic Devices: A New Benchmark Dataset and a Multi-Task Learning based {QA} Framework",
    author = "Nandy, Abhilash  and
      Sharma, Soumya  and
      Maddhashiya, Shubham  and
      Sachdeva, Kapil  and
      Goyal, Pawan  and
      Ganguly, NIloy",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.392",
    doi = "10.18653/v1/2021.findings-emnlp.392",
    pages = "4600--4609",
    abstract = "Answering questions asked from instructional corpora such as E-manuals, recipe books, etc., has been far less studied than open-domain factoid context-based question answering. This can be primarily attributed to the absence of standard benchmark datasets. In this paper, we meticulously create a large amount of data connected with E-manuals and develop a suitable algorithm to exploit it. We collect E-Manual Corpus, a huge corpus of 307,957 E-manuals, and pretrain RoBERTa on this large corpus. We create various benchmark QA datasets which include question answer pairs curated by experts based upon two E-manuals, real user questions from Community Question Answering Forum pertaining to E-manuals etc. We introduce EMQAP (E-Manual Question Answering Pipeline) that answers questions pertaining to electronics devices. Built upon the pretrained RoBERTa, it harbors a supervised multi-task learning framework which efficiently performs the dual tasks of identifying the section in the E-manual where the answer can be found and the exact answer span within that section. For E-Manual annotated question-answer pairs, we show an improvement of about 40{\%} in ROUGE-L F1 scores over most competitive baseline. We perform a detailed ablation study and establish the versatility of EMQAP across different circumstances. The code and datasets are shared at https://github.com/abhi1nandy2/EMNLP-2021-Findings, and the corresponding project website is https://sites.google.com/view/emanualqa/home.",
}
```
