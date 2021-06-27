## Evaluation

### Files

1. `smd.py` - contains helper functions for calculating S+WMS score
2. `metrics.py` - contains functions for calculating ROUGE-L (P, R, F1), Exact Match, and S+WMS scores

> For obtaining S+WMS score, After running the `get_swms` function, a file `input_swms.tsv` is obtained. After this, this needs to be run -

```
python smd.py input_swms.tsv glove s+wms
```
> NOTE: The ROUGE-L (P, R, F1) and S+WMS scores are obtained by averaging over respective scores on individual samples.
