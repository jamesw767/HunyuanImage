#!/bin/bash
# Run HunyuanImage-3.0 Quantized Model (48GB, fits on single 96GB GPU)

# Activate virtual environment
source /media/james/DataDrive/jamesw767/Hun3d/hunyuan_env/bin/activate

# Set CUDA device to GPU 0
export CUDA_VISIBLE_DEVICES=0

# Generate image using quantized model
python3 -c "
import torch
from transformers import AutoModelForCausalLM
from sdnq import SDNQConfig

model_id = '/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage3-SDNQ'
print('Loading quantized model...')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation='sdpa',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    moe_impl='eager',
)
model.load_tokenizer(model_id)

prompt = '''$1'''
print(f'Generating image for: {prompt}')
image = model.generate_image(prompt=prompt, stream=True, diff_infer_steps=20)
output_path = '/media/james/DataDrive/jamesw767/Hun3d/output.png'
image.save(output_path)
print(f'Image saved to {output_path}')
"
