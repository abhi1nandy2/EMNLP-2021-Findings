## Multi-Task Learning (MTL)

### Files

(same `TFIDF+T5` used before MTL as in case of other variants. The outputs given by `TFIDF+T5` serve as input for MTL)
1. `MTL_utils.py` - contains auxiliary helper functions
2. `mtl_data.py` - to convert QA data into `csv` data format (columns - `question`, `section` or `answer` depending upon whether it is for SR or AR,`label`) that goes as input to the **SR** and **AR** modules.
3. `mtl_model.py` - contains the class of the MTL model architecture, used in `mtl_ir_rc.py`
4. `mtl_sr_ar.py` - fine tune the MTL framework
