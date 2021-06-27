## Data Analysis

### Files/Folders

1. `EManual_vs_original_roberta_embeddings` - contains 100 word embeddings each for the `RoBERTa` off-the-shelf-model and `RoBERTa` pre-trained on E-Manuals
2. `nearest_neighbours_embeddings.ipynb` - Jupyter Notebook to view these embeddings (and top-K nearest neighbours) in a `TensorBoard` visualization tool. (Open in [COLAB](https://colab.research.google.com/))

> Check out the two models hosted at the anonymous link - https://gradio.app/g/anon-submission2020/Anonymous-Submission-EMNLP2021
![](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/blob/main/data_analysis/emanual_vs_orig_roberta.png)
> This screenshot shows the format of the masked input text to each of the models. When input is `Each <mask> card has a phone number associated with it.`, `RoBERTa` pre-trained on E-Manuals gives the most likely inference of `<mask>` to be `SIM` (in line with E-Manuals), while `RoBERTa` off-the-shelf-model infers it as `credit` with maximum probability (not related to E-Manuals in general).
![](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/blob/main/data_analysis/emanual_vs_orig_roberta_2.png)
> Another example of the differences in masked language model inference between the two models. The model trained on E-Manuals answers it as `CPU`, which seems to be much more specific than `This` output by the other model.

This is a GIF for that visualization -
![](https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/blob/main/data_analysis/nearest_neighbours.gif)


3. `Sunburst_plots.ipynb` - to make the distribution of first three tokens of the questions present in any dataset.

An interactive Sunburst plot for the distribution is present at [this link](https://htmlpreview.github.io/?https://github.com/anon-submission2020/Anonymous-Submission-EMNLP2021/blob/main/data_analysis/sunburst_figure.html)
