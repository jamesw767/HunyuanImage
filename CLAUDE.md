# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HunyuanImage-3.0 is a local AI image generation system built around Tencent's 80B MoE quantized model, with Ollama LLM integration for intelligent prompt enhancement. It provides a Gradio web UI for text-to-image generation running entirely on local GPU.

## Quick Start Commands

```bash
# Launch the Web UI (recommended)
./launch_ui.sh

# Or with more options/interactive start
./start_ui.sh

# Run directly with Python
source hunyuan_env/bin/activate
CUDA_VISIBLE_DEVICES=1 python hunyuan_ui.py
```

## Architecture

### Core Components

- **hunyuan_ui.py** - Main Gradio web interface (~2200 lines). Handles single image generation, batch processing, wildcards, gallery management, and all UI interactions. Loads the quantized HunyuanImage3-SDNQ model on startup.

- **ollama_prompts.py** - Prompt enhancement via local Ollama LLMs. Contains `OllamaClient` for API communication and `PromptEnhancer` for the 5-part cinematographic prompt formula (subject, quality, composition, lighting, style).

- **ollama_manager.py** - `OllamaManager` class for start/stop/status of Ollama server, plus model management (pull, delete, list).

- **prompt_generator.py** - `PromptGenerator` class for themed prompt generation, character consistency, style fusion, and story sequences via Ollama.

- **wildcard_utils.py** - `WildcardManager` for `[wildcard]` syntax in prompts. Loads from `wildcards.json` and supports nested wildcard resolution.

- **batch_generator.py** - CLI batch processor with SQLite queue (`batch_queue.db`), resume capability, and progress tracking.

### Data Flow

1. User enters prompt in Gradio UI
2. Optional: Wildcards resolved via `WildcardManager`
3. Optional: Prompt enhanced via `PromptEnhancer` (Ollama)
4. Style preset suffix appended
5. Image generated via quantized HunyuanImage model
6. Result saved to `outputs/` with JSON config sidecar

### Key Directories

- `HunyuanImage3-SDNQ/` - Quantized model weights (uint4-svd-r32)
- `HunyuanImage-3.0/` - Original repo with `sdnq` module
- `hunyuan_env/` - Python virtual environment
- `outputs/` - Generated images and `batches/` subdirectory
- `sd-wildcards/` - Source data for wildcards.json

## GPU Configuration

**Critical**: This system uses a dual-GPU setup with reversed CUDA ordering:

```bash
# nvidia-smi shows: GPU 0 = Blackwell (96GB), GPU 1 = Other
# CUDA/PyTorch sees: GPU 0 = Other, GPU 1 = Blackwell
export CUDA_VISIBLE_DEVICES=1  # Use the Blackwell GPU
```

The main UI sets this automatically in `hunyuan_ui.py:2168`.

## Key Technical Details

- Model: HunyuanImage-3.0 quantized with SDNQ (uint4-svd-r32)
- Loads via transformers with `trust_remote_code=True`
- Uses `moe_impl="eager"` for MoE layers
- Generation method: `model.generate_image(prompt, seed, image_size, diff_infer_steps, stream=True)`

## CLI Tools

```bash
# Ollama management
python ollama_manager.py status|start|stop|list|pull <model>|recommended

# Prompt enhancement
python ollama_prompts.py "your prompt" --model qwen2.5:7b-instruct

# Prompt generation
python prompt_generator.py "cyberpunk city" --count 10 --style cinematic

# Batch processing
python batch_generator.py create prompts.txt --enhance
python batch_generator.py run [batch_id]
python batch_generator.py list
```

## Ollama Models

Default: `qwen2.5:7b-instruct` (4.7GB)
Also configured: `magistral:24b`, `qwen3-next:80b`

Server runs at `http://localhost:11434`. Binary at `/home/james/.local/bin/ollama`.

## Wildcard System

Prompts can contain `[wildcard]` tags that get randomly resolved:
```
"A [animal] in a [landscape]" → "A tiger in a forest"
```

Categories are prefixed (e.g., `artist-famous`, `color-warm`). Supports nested resolution.

Combined wildcards with `+` syntax:
```
"A [color+animal]" → "A golden dragon"
```

## HunyuanImage 3.0 Prompting Guide

See **HUNYUAN_GUIDE.md** for comprehensive documentation on:
- Why this model is 80B parameters and what advantages that provides
- How prompting differs from Stable Diffusion and FLUX
- Recommended prompt structure and techniques
- Guidance Scale (CFG) settings
- Text rendering tips
- Best practices for getting optimal results

### Quick Prompting Tips

1. **Write prose, not keywords** - Full sentences work better than comma-separated lists
2. **Use quotes for text rendering** - `with the text "HELLO" in bold serif font`
3. **No bracket emphasis** - Unlike SD, `(word:1.4)` syntax does NOT work
4. **Embed negatives in prompt** - "no watermark, no text" works better than negative prompt field
5. **Be specific** - "85mm lens at f/2.8, golden hour lighting" beats "professional photo"
6. **Define relationships** - Explicitly state how subjects interact spatially
