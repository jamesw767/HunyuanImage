#!/bin/bash
# Run HunyuanImage-3.0 on single GPU (RTX PRO 6000 Blackwell 96GB)

# Activate virtual environment
source /media/james/DataDrive/jamesw767/Hun3d/hunyuan_env/bin/activate

# Set CUDA device to GPU 0 (the 96GB Blackwell)
export CUDA_VISIBLE_DEVICES=0

# Add HunyuanImage-3.0 to Python path
export PYTHONPATH=/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage-3.0:$PYTHONPATH

# Run generation
# --attn-impl: sdpa (PyTorch native, works without FlashAttention)
# --rewrite 0: Disable DeepSeek prompt rewriting (requires API key)
# --diff-infer-steps: 50 is default, can be reduced for faster generation

cd /media/james/DataDrive/jamesw767/Hun3d/HunyuanImage-3.0

python run_image_gen.py \
    --model-id /media/james/DataDrive/jamesw767/Hun3d/HunyuanImage-3 \
    --attn-impl sdpa \
    --rewrite 0 \
    --diff-infer-steps 50 \
    --save /media/james/DataDrive/jamesw767/Hun3d/output.png \
    --prompt "$1"
