#!/usr/bin/env python3
"""
Simple CLI Batch Runner for HunyuanImage-3.0

Runs batch generation from config files or command-line prompts.
Uses round-robin iteration through all prompts.

Usage:
    # Run from saved config with defaults
    python run_batch.py --config outputs/batch_configs/my_config.json

    # Run with custom iterations and resolution
    python run_batch.py --config my_config.json --iterations 40 --resolution 1024x1024

    # Run from prompts file with settings
    python run_batch.py --prompts prompts.txt --iterations 40 --resolution 1536x1024 --steps 35

    # Quick run with inline prompts
    python run_batch.py --prompt "cyberpunk city" --prompt "fantasy forest" --iterations 40
"""

import os
import sys
import gc
import json
import random
import argparse
from datetime import datetime
from pathlib import Path

# Set CUDA device before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

# Add HunyuanImage to path
sys.path.insert(0, str(Path(__file__).parent / "HunyuanImage-3.0"))

BASE_DIR = Path("/media/james/DataDrive/jamesw767/Hun3d")
OUTPUT_DIR = BASE_DIR / "outputs"
BATCHES_DIR = OUTPUT_DIR / "batches"

# Resolution presets (name -> WxH)
RESOLUTIONS = {
    "1:1": "1024x1024",
    "square": "1024x1024",
    "16:9": "1536x864",
    "landscape": "1536x864",
    "9:16": "864x1536",
    "portrait": "864x1536",
    "4:3": "1152x896",
    "3:4": "896x1152",
    "3:2": "1216x832",
    "2:3": "832x1216",
    "21:9": "1680x720",
    "ultrawide": "1680x720",
}

# Quality presets (name -> steps)
QUALITY_STEPS = {
    "draft": 12,
    "fast": 16,
    "standard": 20,
    "high": 35,
    "ultra": 50,
}

# Style presets
STYLE_PRESETS = {
    "none": "",
    "photorealistic": ", photorealistic, hyperrealistic, 8k, highly detailed, professional photography",
    "cinematic": ", cinematic lighting, dramatic atmosphere, movie still, 35mm film grain",
    "anime": ", anime style, vibrant colors, detailed linework, studio quality anime",
    "digital_art": ", digital art, concept art, artstation trending, highly detailed illustration",
    "oil_painting": ", oil painting style, classical art, rich textures, museum quality",
    "fantasy": ", fantasy art, epic fantasy, magical atmosphere, detailed fantasy illustration",
    "3d_render": ", 3D render, octane render, unreal engine 5, high quality CGI",
}


def load_model():
    """Load the HunyuanImage model."""
    from transformers import AutoModelForCausalLM

    model_id = str(BASE_DIR / "HunyuanImage3-SDNQ")
    print(f"Loading model from {model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        moe_impl="eager",
    )
    model.load_tokenizer(model_id)
    print("Model loaded!")
    return model


def load_config(config_path: str) -> dict:
    """Load batch config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_prompts_file(prompts_path: str) -> list:
    """Load prompts from text file (one per line)."""
    prompts = []
    with open(prompts_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


def resolve_resolution(res: str) -> str:
    """Resolve resolution from name or WxH format."""
    res_lower = res.lower().replace(' ', '')
    if res_lower in RESOLUTIONS:
        return RESOLUTIONS[res_lower]
    # Check if it's already WxH format
    if 'x' in res and res.replace('x', '').isdigit():
        return res
    # Default
    return "1024x1024"


def resolve_steps(quality: str) -> int:
    """Resolve steps from quality preset name or number."""
    if isinstance(quality, int):
        return quality
    quality_lower = str(quality).lower()
    if quality_lower in QUALITY_STEPS:
        return QUALITY_STEPS[quality_lower]
    # Try to extract from string like "High Quality (35 steps)"
    if '(' in quality_lower and 'step' in quality_lower:
        import re
        match = re.search(r'(\d+)\s*step', quality_lower)
        if match:
            return int(match.group(1))
    # Try as integer
    try:
        return int(quality)
    except:
        return 20


def init_enhancer(model_name: str):
    """Initialize Ollama prompt enhancer."""
    try:
        from ollama_prompts import PromptEnhancer
        return PromptEnhancer(model=model_name)
    except Exception as e:
        print(f"Warning: Could not initialize enhancer: {e}")
        return None


def init_wildcards():
    """Initialize wildcard manager."""
    try:
        from wildcard_utils import WildcardManager
        return WildcardManager(json_path=BASE_DIR / "wildcards.json")
    except Exception as e:
        print(f"Warning: Could not initialize wildcards: {e}")
        return None


def run_batch(
    prompts: list,
    iterations: int = 40,
    resolution: str = "1024x1024",
    steps: int = 20,
    styles: list = None,
    enhance: bool = False,
    ollama_model: str = "qwen2.5:7b-instruct",
    ollama_length: str = "medium",
    ollama_complexity: str = "detailed",
    batch_name: str = "cli_batch",
    negative_prompt: str = "",
):
    """Run batch generation with round-robin iteration."""

    if not prompts:
        print("Error: No prompts provided")
        return

    if styles is None:
        styles = ["none"]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in batch_name[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    batch_dir = BATCHES_DIR / f"{safe_name}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Batch: {batch_name}")
    print(f"Output: {batch_dir}")
    print(f"Prompts: {len(prompts)}")
    print(f"Styles: {styles}")
    print(f"Iterations: {iterations}")
    print(f"Resolution: {resolution}")
    print(f"Steps: {steps}")
    print(f"Enhance: {enhance}")
    print(f"{'='*60}\n")

    # Calculate total images
    total_images = len(prompts) * len(styles) * iterations
    print(f"Total images to generate: {total_images}")

    # Load model
    model = load_model()

    # Initialize enhancer if needed
    enhancer = None
    if enhance:
        enhancer = init_enhancer(ollama_model)
        if enhancer:
            print(f"Prompt enhancement enabled: {ollama_model}")

    # Initialize wildcards
    wildcard_manager = init_wildcards()
    if wildcard_manager:
        print(f"Wildcards loaded: {len(wildcard_manager.data)} categories")

    # Length/complexity options for random selection
    length_choices = ["minimal", "short", "medium", "long", "detailed", "cinematic", "experimental"]
    complexity_choices = ["simple", "moderate", "detailed", "complex", "cinematic", "experimental"]

    count = 0

    # Round-robin iteration: cycle through all prompts/styles before next iteration
    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")

        for prompt_idx, base_prompt in enumerate(prompts):
            for style in styles:
                count += 1

                # Build prompt
                full_prompt = base_prompt

                # Apply style
                style_suffix = STYLE_PRESETS.get(style.lower(), "")
                if style_suffix:
                    full_prompt += style_suffix

                # Enhance with Ollama
                if enhancer:
                    try:
                        # Handle random length/complexity
                        actual_length = ollama_length
                        actual_complexity = ollama_complexity

                        if str(ollama_length).lower() == "random":
                            actual_length = random.choice(length_choices)
                        if str(ollama_complexity).lower() == "random":
                            actual_complexity = random.choice(complexity_choices)

                        full_prompt = enhancer.enhance(
                            full_prompt,
                            length=actual_length,
                            complexity=actual_complexity,
                            model=ollama_model
                        )
                    except Exception as e:
                        print(f"  Enhancement failed: {e}")

                # Resolve wildcards
                if wildcard_manager:
                    try:
                        full_prompt = wildcard_manager.process(full_prompt)
                    except:
                        pass

                # Generate seed
                seed = random.randint(0, 2**31 - 1)

                print(f"[{count}/{total_images}] Iter {iteration+1}, Prompt {prompt_idx+1}, Style: {style}")
                print(f"  Prompt: {full_prompt[:80]}...")

                try:
                    image = model.generate_image(
                        full_prompt,
                        seed,
                        resolution,
                        stream=True,
                        diff_infer_steps=steps
                    )

                    if image:
                        filename = f"{count:04d}_{seed}_{style.replace(' ', '_')[:20]}.png"
                        filepath = batch_dir / filename
                        image.save(str(filepath))

                        # Save config sidecar
                        config = {
                            "prompt": base_prompt,
                            "full_prompt": full_prompt,
                            "style": style,
                            "seed": seed,
                            "steps": steps,
                            "resolution": resolution,
                            "iteration": iteration + 1,
                        }
                        config_path = filepath.with_suffix('.json')
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)

                        print(f"  Saved: {filename}")

                    # Cleanup
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"  Error: {e}")

    print(f"\n{'='*60}")
    print(f"Batch complete! Generated {count} images")
    print(f"Output: {batch_dir}")
    print(f"{'='*60}")

    # Save batch summary
    summary = {
        "batch_name": batch_name,
        "completed_at": datetime.now().isoformat(),
        "total_images": count,
        "prompts": prompts,
        "styles": styles,
        "iterations": iterations,
        "resolution": resolution,
        "steps": steps,
        "enhance": enhance,
    }
    with open(batch_dir / "batch_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="CLI Batch Runner for HunyuanImage-3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from saved config
  python run_batch.py --config outputs/batch_configs/my_config.json

  # Run with custom settings
  python run_batch.py --config my_config.json --iterations 40 --resolution 1536x1024

  # Run from prompts file
  python run_batch.py --prompts prompts.txt --iterations 40 --steps 35

  # Quick inline prompts
  python run_batch.py --prompt "cyberpunk city" --prompt "fantasy forest" -i 40

Resolution presets: square, landscape, portrait, 16:9, 9:16, 4:3, 3:4, ultrawide
Quality presets: draft (12), fast (16), standard (20), high (35), ultra (50)
        """
    )

    # Input sources
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--config', '-c', help='Load from batch config JSON file')
    input_group.add_argument('--prompts', '-p', help='Load prompts from text file (one per line)')
    input_group.add_argument('--prompt', action='append', help='Add inline prompt (can use multiple times)')

    # Generation settings
    gen_group = parser.add_argument_group('Generation')
    gen_group.add_argument('--iterations', '-i', type=int, default=40, help='Iterations per prompt (default: 40)')
    gen_group.add_argument('--resolution', '-r', default='1024x1024', help='Resolution (e.g., 1024x1024, landscape, 16:9)')
    gen_group.add_argument('--steps', '-s', type=int, help='Inference steps (default: from config or 20)')
    gen_group.add_argument('--quality', '-q', help='Quality preset (draft/fast/standard/high/ultra)')
    gen_group.add_argument('--style', action='append', help='Style preset (can use multiple)')
    gen_group.add_argument('--name', '-n', default='cli_batch', help='Batch name')
    gen_group.add_argument('--negative', help='Negative prompt')

    # Enhancement
    enhance_group = parser.add_argument_group('Enhancement')
    enhance_group.add_argument('--enhance', '-e', action='store_true', help='Enhance prompts with Ollama')
    enhance_group.add_argument('--ollama-model', default='qwen2.5:7b-instruct', help='Ollama model')
    enhance_group.add_argument('--ollama-length', default='medium', help='Enhancement length (or "random")')
    enhance_group.add_argument('--ollama-complexity', default='detailed', help='Enhancement complexity (or "random")')

    args = parser.parse_args()

    # Collect prompts from various sources
    prompts = []
    styles = args.style or ["none"]
    iterations = args.iterations
    resolution = args.resolution
    steps = args.steps
    batch_name = args.name
    negative_prompt = args.negative or ""
    enhance = args.enhance
    ollama_model = args.ollama_model
    ollama_length = args.ollama_length
    ollama_complexity = args.ollama_complexity

    # Load from config file
    if args.config:
        config = load_config(args.config)

        # Extract prompts from themes field
        themes = config.get('themes', '')
        if themes:
            prompts = [t.strip() for t in themes.split('\n') if t.strip()]

        # Use config values as defaults (CLI args override)
        if not args.style:
            styles = config.get('styles', ['none'])
        if args.iterations == 40:  # default
            iterations = config.get('variations_per_theme', 40)
        if not args.resolution or args.resolution == '1024x1024':
            aspect = config.get('aspect_ratio', '1:1 (Square)')
            # Map aspect ratio names to resolutions
            aspect_map = {
                '1:1 (Square)': '1024x1024',
                '16:9 (Landscape)': '1536x864',
                '9:16 (Portrait)': '864x1536',
                '4:3': '1152x896',
                '3:4': '896x1152',
                '3:2': '1216x832',
                '2:3': '832x1216',
                '21:9 (Ultrawide)': '1680x720',
            }
            resolution = aspect_map.get(aspect, '1024x1024')
        if not args.steps:
            quality = config.get('quality_preset', 'Standard')
            steps = resolve_steps(quality)
        if args.name == 'cli_batch':
            batch_name = config.get('batch_name', 'cli_batch')
        if not args.negative:
            negative_prompt = config.get('negative_prompt', '')
        if not args.enhance:
            enhance = config.get('enhance_prompts', False)
        if args.ollama_model == 'qwen2.5:7b-instruct':
            ollama_model = config.get('ollama_model', 'qwen2.5:7b-instruct')
        if args.ollama_length == 'medium':
            ollama_length = config.get('ollama_length', 'medium')
        if args.ollama_complexity == 'detailed':
            ollama_complexity = config.get('ollama_complexity', 'detailed')

    # Load from prompts file
    if args.prompts:
        prompts.extend(load_prompts_file(args.prompts))

    # Add inline prompts
    if args.prompt:
        prompts.extend(args.prompt)

    # Resolve quality preset to steps
    if args.quality:
        steps = resolve_steps(args.quality)
    if steps is None:
        steps = 20

    # Resolve resolution
    resolution = resolve_resolution(resolution)

    if not prompts:
        print("Error: No prompts provided. Use --config, --prompts, or --prompt")
        parser.print_help()
        return

    # Run the batch
    run_batch(
        prompts=prompts,
        iterations=iterations,
        resolution=resolution,
        steps=steps,
        styles=styles,
        enhance=enhance,
        ollama_model=ollama_model,
        ollama_length=ollama_length,
        ollama_complexity=ollama_complexity,
        batch_name=batch_name,
        negative_prompt=negative_prompt,
    )


if __name__ == "__main__":
    main()
