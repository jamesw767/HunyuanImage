# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is This Project?

**HunyuanImage-3.0** is a local AI image generation system - similar to Midjourney or DALL-E, but running entirely on your own hardware. Type a text description, get a high-quality image.

**Why this exists:** To have a powerful, private, no-subscription image generation capability without relying on cloud services.

**Key capabilities:**
- Generate photorealistic images, artwork, illustrations from text prompts
- Multiple aspect ratios (square, portrait, landscape, ultrawide)
- Reproducible results using seed values
- ~1.5 minutes per 1024x1024 image on single GPU

## Project Overview

HunyuanImage-3.0 is Tencent's native multimodal text-to-image generation model. It's an 80B parameter Mixture of Experts (MoE) model with 13B active parameters per token, using an autoregressive framework rather than DiT-based architecture.

## Installation History

This environment was set up in December 2024. Here's what was installed:

### Step 1: Create Virtual Environment
```bash
python3 -m venv /media/james/DataDrive/jamesw767/Hun3d/hunyuan_env
source hunyuan_env/bin/activate
```

### Step 2: Clone the Repository
```bash
git clone https://github.com/Tencent/HunyuanImage-3.0.git
```

### Step 3: Install Dependencies
```bash
# IMPORTANT: Use cu128 for Blackwell GPU (CUDA 12.8 required)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate gradio pillow
pip install -e HunyuanImage-3.0/
```

### Step 4: Download Model Weights
**Full model (158GB, requires 240GB+ VRAM):**
```bash
huggingface-cli download tencent/HunyuanImage-3 --local-dir HunyuanImage-3
```

**Quantized model (48GB, fits single 96GB GPU) - RECOMMENDED:**
```bash
pip install sdnq
huggingface-cli download Disty0/HunyuanImage3-SDNQ-uint4-svd-r32 --local-dir HunyuanImage3-SDNQ
```

### Step 5: Create Convenience Scripts
The `run_quantized.sh`, `start_ui.sh`, and `hunyuan_ui.py` files were created for easy access.

## Directory Structure

- `HunyuanImage-3.0/` - Main source code repository
  - `hunyuan_image_3/` - Core model implementation (model architecture, config, tokenizer, image processing)
  - `app/` - Gradio web interface
  - `PE/` - Prompt engineering (DeepSeek prompt rewriting system prompts)
  - `vllm_infer/` - vLLM backend for faster inference
- `HunyuanImage-3/` - Downloaded full model weights (158GB)
- `HunyuanImage3-SDNQ/` - 4-bit quantized model (48GB) - recommended for single GPU
- `hunyuan_env/` - Python virtual environment
- `run_hunyuan.sh` - Convenience script for single-GPU inference
- `run_quantized.sh` - Convenience script for quantized model (recommended)
- `hunyuan_ui.py` - Gradio web UI for image generation (with Ollama integration)
- `start_ui.sh` - Launcher script for the web UI
- `outputs/` - Directory where generated images are saved
- **Ollama Integration:**
  - `ollama_manager.py` - Server management (start/stop/models)
  - `ollama.sh` - CLI for server control
  - `ollama_prompts.py` - Core Ollama client for prompt enhancement
  - `batch_generator.py` - Batch processing with queue management
  - `prompt_generator.py` - Creative prompt generation from themes
  - `enhance_prompt.sh` - CLI for single prompt enhancement
  - `generate_prompts.sh` - CLI for themed prompt generation
  - `batch_run.sh` - CLI for batch image generation
  - `ollama_config.json` - Configuration for Ollama integration

## Common Commands

### Environment Setup
```bash
source /media/james/DataDrive/jamesw767/Hun3d/hunyuan_env/bin/activate
export PYTHONPATH=/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage-3.0:$PYTHONPATH
```

### Launch Web UI (Recommended)
```bash
./start_ui.sh
```
Opens a Gradio web interface at http://localhost:7860 with:
- Visual prompt input with examples
- Aspect ratio presets (1:1, 16:9, 9:16, etc.)
- Inference steps and seed controls
- Auto-save to `outputs/` directory
- Image gallery of recent generations

### Generate Image (Command Line)
```bash
./run_quantized.sh "your prompt here"
```
The quantized model (48GB) fits on a single 96GB GPU and generates 1024x1024 images in ~1.5 minutes.

### Generate Image (Full Model - Requires 240GB+ VRAM)
```bash
./run_hunyuan.sh "your prompt here"
```

### Generate Image (Direct)
```bash
cd HunyuanImage-3.0
python run_image_gen.py \
    --model-id /media/james/DataDrive/jamesw767/Hun3d/HunyuanImage-3 \
    --attn-impl sdpa \
    --rewrite 0 \
    --prompt "A brown and white dog running on grass"
```

### Launch Gradio Web UI
```bash
cd HunyuanImage-3.0
export MODEL_ID="/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage-3"
export GPUS="0"
sh run_app.sh
```

### vLLM Server (Faster Inference)
```bash
sh vllm_infer/run_vllm_server.sh /path/to/model
python vllm_infer/openai_client.py --prompt "your prompt"
```

## Key CLI Arguments (run_image_gen.py)

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | Input text prompt | Required |
| `--model-id` | Path to model weights | Required |
| `--attn-impl` | `sdpa` or `flash_attention_2` | `sdpa` |
| `--moe-impl` | `eager` or `flashinfer` | `eager` |
| `--rewrite` | Enable DeepSeek prompt rewriting (requires API key) | `1` |
| `--diff-infer-steps` | Diffusion inference steps | `50` |
| `--image-size` | `auto`, `WxH`, or `W:H` | `auto` |
| `--seed` | Random seed for reproducibility | None |

## System Requirements

### CRITICAL: Blackwell GPU Requirements (RTX PRO 6000 Blackwell)
The RTX PRO 6000 Blackwell GPU requires strict software versions. Many other image generators fail on Blackwell because they don't support these requirements:

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| **CUDA** | 12.8+ | Blackwell architecture requires CUDA 12.8 minimum |
| **PyTorch** | 2.7+ | Must be built with CUDA 12.8 support |
| **Python** | 3.12+ | Required for PyTorch 2.7 compatibility |

**Why HunyuanImage-3.0 works on Blackwell:** It's one of the few generators actively maintained with CUDA 12.8/PyTorch 2.7+ support.

**Verify your environment before running:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
nvcc --version  # Should show 12.8+
```

### Full Model (158GB)
- GPU Memory: Minimum 3x80GB (240GB total), 4x80GB recommended
- CUDA 12.8+, PyTorch 2.7+, Python 3.12+

### Quantized Model (48GB) - Recommended for Single GPU
- GPU Memory: Single 96GB GPU (RTX PRO 6000 Blackwell)
- CUDA 12.8+, PyTorch 2.7+ (strict requirements for Blackwell)
- Generates 1024x1024 images in ~1.5 minutes with 20 inference steps
- Uses SDNQ uint4 quantization from [Disty0/HunyuanImage3-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/HunyuanImage3-SDNQ-uint4-svd-r32)

### GPU Ordering Note (IMPORTANT)
nvidia-smi and CUDA/PyTorch have REVERSED GPU ordering on this system:

| nvidia-smi | CUDA/PyTorch | GPU |
|------------|--------------|-----|
| GPU 0 | cuda:1 / CUDA_VISIBLE_DEVICES=1 | RTX PRO 6000 Blackwell (96GB) |
| GPU 1 | cuda:0 / CUDA_VISIBLE_DEVICES=0 | RTX 5000 Ada |

To use the Blackwell GPU, set `CUDA_VISIBLE_DEVICES=1` (not 0).

## Performance Optimizations

For 3x faster inference, install:
```bash
pip install flash-attn==2.8.3 --no-build-isolation  # Then use --attn-impl flash_attention_2
pip install flashinfer-python                        # Then use --moe-impl flashinfer
```

Note: First inference with FlashInfer takes ~10 minutes for kernel compilation.

## Ollama Integration (Added December 2024)

Local LLM-powered prompt enhancement and batch processing using Ollama.

### Available Ollama Models
| Model | Size | Best Use |
|-------|------|----------|
| `qwen2.5:7b-instruct` | 4.7GB | Fast prompt enhancement (default) |
| `magistral:24b` | 14GB | Creative prompt generation |
| `qwen3-next:80b` | 50GB | Highest quality prompts |

### Ollama Server Management

**Using the management script:**
```bash
./ollama.sh status      # Show server status and installed models
./ollama.sh start       # Start Ollama server
./ollama.sh stop        # Stop Ollama server
./ollama.sh restart     # Restart Ollama server
./ollama.sh list        # List installed models
./ollama.sh recommended # Show recommended models for prompt generation
```

**Installing new models:**
```bash
./ollama.sh pull llama3.2:3b    # Fast, lightweight (2GB)
./ollama.sh pull mistral:7b     # Good creative writing (4GB)
./ollama.sh pull gemma2:9b      # Balanced performance (5GB)
./ollama.sh pull qwen2.5:14b    # Higher quality (9GB)
```

**Removing models:**
```bash
./ollama.sh delete <model-name>
```

### Ollama Prompt Commands

**Enhance a single prompt:**
```bash
./enhance_prompt.sh "a dog running in a field"
./enhance_prompt.sh "mountain landscape" --style cinematic
```

**Generate themed prompts:**
```bash
./generate_prompts.sh "cyberpunk cities" 10
./generate_prompts.sh "underwater scenes" 20 --style cinematic --output batch.json
```

**Batch processing:**
```bash
# Create batch from file
./batch_run.sh create prompts.csv
./batch_run.sh create prompts.json --enhance

# Run batch
./batch_run.sh run
./batch_run.sh run batch_20251230_123456

# List batches
./batch_run.sh list

# Resume interrupted batch
./batch_run.sh resume batch_20251230_123456
```

**Pipeline example (generate prompts → batch images):**
```bash
./generate_prompts.sh "fantasy landscapes" 50 -o fantasy.json
./batch_run.sh create fantasy.json --enhance
./batch_run.sh run
```

### Web UI Ollama Features
The web UI includes:
- **Ollama tab** with:
  - Server controls (Start/Stop/Refresh)
  - "Enhance prompts with Ollama" toggle
  - Model selector dropdown (auto-populated from installed models)
  - Install/Remove models interface
- **Prompt Generator accordion**: Generate themed prompts directly in the UI

### Batch Input File Formats

**CSV format:**
```csv
prompt,style,seed,aspect_ratio,steps
"a serene mountain lake",cinematic,,16:9,20
"cyberpunk city street",realistic,12345,1:1,30
```

**JSON format:**
```json
{
  "batch_name": "my_batch",
  "prompts": [
    {"prompt": "...", "style": "cinematic"},
    {"prompt": "...", "seed": 42, "steps": 30}
  ]
}
```

**Text format (one prompt per line):**
```
A mountain landscape at sunset
A futuristic city with flying cars
An underwater temple with bioluminescent fish
```

## Architecture Notes

- Model uses autoregressive framework (not DiT-based like Stable Diffusion)
- 64 experts total, 13B parameters activated per token
- Supports automatic resolution prediction based on prompt content
- Core model class: `HunyuanImage3ForCausalMM` in `hunyuan_image_3/hunyuan.py`
- Image processor: `HunyuanImageProcessor` in `hunyuan_image_3/image_processor.py`
- VAE decoder: `AutoencoderKL3D` in `hunyuan_image_3/autoencoder_kl_3d.py`
