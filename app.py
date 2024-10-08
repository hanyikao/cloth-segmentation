import PIL
import torch
import gradio as gr
from infer import load_seg_model, get_palette
from process import generate_mask


device = 'gpu'



def initialize_and_load_models():

    checkpoint_path = 'trained_checkpoint/cloth_segm_u2net_latest.pth'
    net = load_seg_model(checkpoint_path, device=device)    

    return net

net = initialize_and_load_models()
palette = get_palette(4)


def run(img):

    cloth_seg = generate_mask(img, net=net, palette=palette, device=device)
    return cloth_seg

# Define input and output interfaces
input_image = gr.inputs.Image(label="Input Image", type="pil")

# Define the Gradio interface
cloth_seg_image = gr.outputs.Image(label="Cloth Segmentation", type="pil")

title = "Demo for Cloth Segmentation"
description = "An app for Cloth Segmentation"
inputs = [input_image]
outputs = [cloth_seg_image]


gr.Interface(fn=run, inputs=inputs, outputs=outputs, title=title, description=description).launch(share=True)