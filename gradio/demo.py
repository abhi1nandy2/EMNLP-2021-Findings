import gradio as gr
from gradio.mix import Parallel, Series

title = "Masked language Model Demo"
description = "demo for masked laguage modelling. To use it, simply add your text with a <mask> token for the model to infer, or click one of the examples to load them."

examples = [
    ['Press the <mask> in the remote.,'],
    ["Attach the charger to the <mask>"]
]

io1 = gr.Interface.load('huggingface/roberta-base', inputs=gr.inputs.Textbox(lines=2, label="Input Text"),title=title,description=description, examples=examples)#, api_key=api_key)
io2 = gr.Interface.load('huggingface/AnonymousSub/EManuals_Roberta', inputs=gr.inputs.Textbox(lines=2, label="Input Text"),title=title,description=description, examples=examples)#, api_key=api_key)

Parallel(io1, io2).launch()
