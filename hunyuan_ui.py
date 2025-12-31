#!/usr/bin/env python3
"""
HunyuanImage-3.0 Quantized Model UI
A comprehensive Gradio interface for text-to-image generation.
Now with Ollama LLM integration for prompt enhancement and batch processing.
"""

import os
import sys
import time
import random
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
import gradio as gr
from PIL import Image

# Global model variable
model = None
model_loaded = False

# Ollama integration
ollama_available = False
ollama_manager = None
try:
    from ollama_prompts import PromptEnhancer, OllamaClient
    from prompt_generator import PromptGenerator
    from ollama_manager import OllamaManager
    ollama_available = True
    ollama_manager = OllamaManager()
except ImportError:
    pass

# Wildcard integration
wildcard_available = False
wildcard_manager = None
try:
    from wildcard_utils import WildcardManager, insert_wildcard, preview_wildcard
    wildcard_manager = WildcardManager(
        json_path=Path(__file__).parent / "wildcards.json"
    )
    wildcard_available = True
except ImportError:
    pass

# Ollama globals
ollama_enhancer = None
ollama_generator = None
OLLAMA_MODELS = ["qwen2.5:7b-instruct", "magistral:24b", "qwen3-next:80b"]

# Output directory
OUTPUT_DIR = Path("/media/james/DataDrive/jamesw767/Hun3d/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# History file for tracking generations
HISTORY_FILE = OUTPUT_DIR / "generation_history.json"

# Supported image sizes
IMAGE_SIZES = [
    "auto",
    "1024x1024",
    "1280x768",
    "768x1280",
    "1152x896",
    "896x1152",
    "1344x768",
    "768x1344",
    "1536x640",
    "640x1536",
]

# Aspect ratio presets
ASPECT_RATIOS = {
    "1:1 (Square)": "1024x1024",
    "16:9 (Landscape)": "1280x768",
    "9:16 (Portrait)": "768x1280",
    "4:3 (Standard)": "1152x896",
    "3:4 (Portrait Standard)": "896x1152",
    "21:9 (Ultrawide)": "1536x640",
    "9:21 (Tall)": "640x1536",
    "Auto (Model decides)": "auto",
}

# Style presets with suffix text
STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", photorealistic, hyperrealistic, 8k, highly detailed, professional photography",
    "Cinematic": ", cinematic lighting, dramatic atmosphere, movie still, 35mm film grain",
    "Anime": ", anime style, vibrant colors, detailed linework, studio quality anime",
    "Digital Art": ", digital art, concept art, artstation trending, highly detailed illustration",
    "Oil Painting": ", oil painting style, classical art, rich textures, museum quality",
    "Watercolor": ", watercolor painting, soft edges, flowing colors, artistic",
    "3D Render": ", 3D render, octane render, unreal engine 5, high quality CGI",
    "Fantasy Art": ", fantasy art, epic fantasy, magical atmosphere, detailed fantasy illustration",
    "Minimalist": ", minimalist style, clean lines, simple composition, elegant",
    "Vintage": ", vintage photograph, retro style, nostalgic, aged film quality",
    "Comic Book": ", comic book style, bold lines, dynamic composition, graphic novel art",
    "Studio Portrait": ", studio portrait photography, professional lighting, sharp focus, bokeh background",
    "Nature Photography": ", national geographic style, nature photography, stunning natural light",
}

# Quality presets
QUALITY_PRESETS = {
    "Draft (Fast)": {"steps": 15, "description": "Quick preview, lower quality"},
    "Standard": {"steps": 20, "description": "Good balance of speed and quality"},
    "High Quality": {"steps": 30, "description": "Better details, slower"},
    "Maximum": {"steps": 50, "description": "Best quality, slowest"},
}


def load_model():
    """Load the quantized HunyuanImage-3.0 model."""
    global model, model_loaded

    if model_loaded:
        return "Model already loaded and ready!"

    try:
        from transformers import AutoModelForCausalLM
        from sdnq import SDNQConfig  # Registers SDNQ into transformers

        model_id = "/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage3-SDNQ"

        yield "Loading quantized HunyuanImage-3.0 model... (this may take 1-2 minutes)"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            moe_impl="eager",
        )
        model.load_tokenizer(model_id)
        model_loaded = True

        yield "Model loaded successfully! Ready to generate images."

    except Exception as e:
        yield f"Error loading model: {str(e)}"


def apply_style(prompt: str, style: str) -> str:
    """Apply a style preset to the prompt."""
    style_suffix = STYLE_PRESETS.get(style, "")
    if style_suffix and not prompt.endswith(style_suffix):
        return prompt.rstrip('.') + style_suffix
    return prompt


def init_ollama(model_name: str = "qwen2.5:7b-instruct"):
    """Initialize Ollama enhancer and generator"""
    global ollama_enhancer, ollama_generator
    if not ollama_available:
        return "Ollama modules not available"
    try:
        ollama_enhancer = PromptEnhancer(model=model_name)
        ollama_generator = PromptGenerator(model=model_name)
        return f"Ollama initialized with {model_name}"
    except Exception as e:
        return f"Ollama init error: {e}"


def enhance_with_ollama(prompt: str, model_name: str, style: str = None,
                        length: str = "medium", complexity: str = "detailed") -> str:
    """Enhance a prompt using Ollama"""
    global ollama_enhancer
    if not ollama_available:
        return prompt, "Ollama not available"
    try:
        if ollama_enhancer is None:
            ollama_enhancer = PromptEnhancer(model=model_name)
        enhanced = ollama_enhancer.enhance(
            prompt,
            style=style if style != "None" else None,
            length=length,
            complexity=complexity
        )
        return enhanced, f"Enhanced ({length}/{complexity})"
    except Exception as e:
        return prompt, f"Enhancement failed: {e}"


def generate_prompts_ollama(theme: str, count: int, model_name: str, style: str = None,
                            length: str = "medium", complexity: str = "detailed") -> List[str]:
    """Generate prompts using Ollama"""
    global ollama_generator
    if not ollama_available:
        return ["Ollama not available"]
    try:
        if ollama_generator is None:
            ollama_generator = PromptGenerator(model=model_name)
        prompts = ollama_generator.generate_themed_prompts(
            theme,
            count=count,
            style=style if style != "None" else None,
            length=length,
            complexity=complexity
        )
        return prompts
    except Exception as e:
        return [f"Error: {e}"]


def check_ollama_status():
    """Check if Ollama is running and return status"""
    if not ollama_available:
        return "Ollama modules not installed"
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            return f"Running - {len(models)} model(s): {', '.join(models[:4])}"
        return "Not responding"
    except:
        return "Stopped"


def get_ollama_models_list():
    """Get list of installed Ollama models for dropdown"""
    if not ollama_available or not ollama_manager:
        return OLLAMA_MODELS
    try:
        models = ollama_manager.list_models()
        if models:
            return [m['name'] for m in models]
    except:
        pass
    return OLLAMA_MODELS


def start_ollama_server():
    """Start the Ollama server"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available", gr.update()
    success, msg = ollama_manager.start()
    models = get_ollama_models_list()
    return msg, gr.update(choices=models, value=models[0] if models else None)


def stop_ollama_server():
    """Stop the Ollama server"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available"
    success, msg = ollama_manager.stop()
    return msg


def pull_ollama_model(model_name: str):
    """Pull/install a new Ollama model"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available", gr.update()

    if not model_name or not model_name.strip():
        return "Please enter a model name", gr.update()

    # Start server if not running
    if not ollama_manager.is_running():
        ollama_manager.start()

    yield f"Pulling {model_name}... (this may take a while)", gr.update()

    def progress_cb(status, pct):
        pass  # Gradio doesn't support real-time updates well in this context

    success, msg = ollama_manager.pull_model(model_name.strip())
    models = get_ollama_models_list()
    yield msg, gr.update(choices=models, value=model_name.strip() if success else (models[0] if models else None))


def delete_ollama_model(model_name: str):
    """Delete an Ollama model"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available", gr.update()

    if not model_name:
        return "No model selected", gr.update()

    success, msg = ollama_manager.delete_model(model_name)
    models = get_ollama_models_list()
    return msg, gr.update(choices=models, value=models[0] if models else None)


# ============================================================================
# BATCH GENERATION SYSTEM
# ============================================================================

# Global batch state
batch_running = False
batch_stop_requested = False
batch_results = []

def calculate_batch_total(themes_text: str, variations_per_theme: int,
                          styles_selected: List[str], images_per_combo: int) -> str:
    """Calculate total images that will be generated"""
    themes = [t.strip() for t in themes_text.strip().split('\n') if t.strip()]
    num_themes = len(themes) if themes else 0
    num_styles = len(styles_selected) if styles_selected else 1

    # Each theme generates variations, each variation gets each style, each combo gets N images
    total_variations = num_themes * variations_per_theme
    total_combos = total_variations * num_styles
    total_images = total_combos * images_per_combo

    # Estimate time (roughly 90 seconds per image)
    est_minutes = (total_images * 90) / 60
    est_hours = est_minutes / 60

    if est_hours >= 1:
        time_str = f"~{est_hours:.1f} hours"
    else:
        time_str = f"~{est_minutes:.0f} minutes"

    return f"""**Batch Summary:**
- Themes/Prompts: {num_themes}
- Variations per theme: {variations_per_theme}
- Styles: {num_styles}
- Images per combination: {images_per_combo}

**Total images to generate: {total_images}**
Estimated time: {time_str}"""


def generate_batch_prompts(themes_text: str, variations_per_theme: int,
                           styles_selected: List[str], ollama_model: str,
                           enhance_prompts: bool,
                           ollama_length: str = "medium",
                           ollama_complexity: str = "detailed") -> List[dict]:
    """Generate all prompt combinations for batch"""
    global ollama_generator, ollama_enhancer

    themes = [t.strip() for t in themes_text.strip().split('\n') if t.strip()]
    if not themes:
        return []

    all_prompts = []

    # Initialize Ollama if needed
    if ollama_available and (variations_per_theme > 1 or enhance_prompts):
        if ollama_generator is None:
            ollama_generator = PromptGenerator(model=ollama_model)
        if ollama_enhancer is None:
            ollama_enhancer = PromptEnhancer(model=ollama_model)

    for theme in themes:
        # Generate variations using Ollama
        if variations_per_theme > 1 and ollama_available and ollama_generator:
            try:
                variations = ollama_generator.generate_themed_prompts(
                    theme, count=variations_per_theme, temperature=0.85,
                    length=ollama_length, complexity=ollama_complexity
                )
            except Exception as e:
                variations = [theme]  # Fall back to original
        else:
            variations = [theme]

        # Enhance each variation if requested
        if enhance_prompts and ollama_available and ollama_enhancer:
            enhanced_variations = []
            for v in variations:
                try:
                    enhanced = ollama_enhancer.enhance(
                        v, temperature=0.7,
                        length=ollama_length, complexity=ollama_complexity
                    )
                    enhanced_variations.append(enhanced)
                except:
                    enhanced_variations.append(v)
            variations = enhanced_variations

        # Create combinations with styles
        styles = styles_selected if styles_selected else ["None"]
        for variation in variations:
            for style in styles:
                all_prompts.append({
                    "prompt": variation,
                    "style": style,
                    "original_theme": theme
                })

    return all_prompts


def run_batch_generation(
    themes_text: str,
    variations_per_theme: int,
    styles_selected: List[str],
    images_per_combo: int,
    ollama_model: str,
    enhance_prompts: bool,
    aspect_ratio: str,
    quality_preset: str,
    random_seeds: bool,
    batch_name: str,
    ollama_length: str = "medium",
    ollama_complexity: str = "detailed",
    progress=gr.Progress()
):
    """Run the batch generation process"""
    global model, model_loaded, batch_running, batch_stop_requested, batch_results

    if not model_loaded:
        yield [], "Please load the model first!", "", []
        return

    if batch_running:
        yield [], "A batch is already running!", "", []
        return

    batch_running = True
    batch_stop_requested = False
    batch_results = []

    try:
        # Create batch output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in batch_name[:20] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        batch_dir = OUTPUT_DIR / "batches" / f"{safe_name}_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        progress(0, desc="Generating prompts with Ollama...")
        yield [], f"Generating prompt variations with Ollama ({ollama_length}/{ollama_complexity})...", "", []

        # Generate all prompt combinations
        prompts = generate_batch_prompts(
            themes_text, variations_per_theme, styles_selected,
            ollama_model, enhance_prompts, ollama_length, ollama_complexity
        )

        if not prompts:
            yield [], "No prompts generated. Please enter at least one theme.", "", []
            batch_running = False
            return

        # Expand for images_per_combo
        all_jobs = []
        for p in prompts:
            for i in range(images_per_combo):
                all_jobs.append({**p, "combo_index": i})

        total_jobs = len(all_jobs)

        # Get settings
        image_size = ASPECT_RATIOS.get(aspect_ratio, "1024x1024")
        steps = QUALITY_PRESETS.get(quality_preset, {}).get("steps", 20)

        # Save batch manifest
        manifest = {
            "batch_name": batch_name,
            "created_at": datetime.now().isoformat(),
            "total_images": total_jobs,
            "settings": {
                "themes": themes_text.split('\n'),
                "variations_per_theme": variations_per_theme,
                "styles": styles_selected,
                "images_per_combo": images_per_combo,
                "aspect_ratio": aspect_ratio,
                "quality": quality_preset,
                "ollama_model": ollama_model,
                "enhance_prompts": enhance_prompts
            },
            "prompts": prompts
        }
        with open(batch_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        generated_images = []

        for idx, job in enumerate(all_jobs):
            if batch_stop_requested:
                yield generated_images, f"Batch stopped at {idx}/{total_jobs}", f"Output: {batch_dir}", generated_images[-12:] if generated_images else []
                break

            prompt = job["prompt"]
            style = job["style"]

            # Generate seed first (needed for wildcard processing)
            seed = random.randint(0, 2**32 - 1) if random_seeds else (idx * 12345) % (2**32)

            # Process wildcards if present
            original_prompt = prompt
            if wildcard_available and wildcard_manager and wildcard_manager.has_wildcards(prompt):
                prompt = wildcard_manager.process_prompt(prompt, seed=seed)

            # Apply style
            styled_prompt = apply_style(prompt, style)

            progress((idx + 1) / total_jobs, desc=f"Generating {idx + 1}/{total_jobs}...")
            yield generated_images, f"Generating {idx + 1}/{total_jobs}: {prompt[:50]}...", f"Output: {batch_dir}", generated_images[-12:] if generated_images else []

            try:
                start_time = time.time()

                image = model.generate_image(
                    prompt=styled_prompt,
                    seed=seed,
                    image_size=image_size,
                    diff_infer_steps=steps,
                    stream=True,
                )

                gen_time = time.time() - start_time

                # Save image
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
                filename = f"{idx+1:04d}_{safe_prompt}_s{seed}.png"
                filepath = batch_dir / filename
                image.save(filepath)

                # Save config JSON for this image
                wildcards_used = (original_prompt != prompt)
                config = {
                    "prompt": original_prompt,
                    "processed_prompt": prompt if wildcards_used else None,
                    "styled_prompt": styled_prompt,
                    "style": style,
                    "seed": seed,
                    "image_size": image_size,
                    "aspect_ratio": aspect_ratio,
                    "steps": steps,
                    "quality_preset": quality_preset,
                    "batch_name": batch_name,
                    "batch_index": idx + 1,
                    "wildcards_used": wildcards_used,
                    "generation_time": gen_time
                }
                save_image_config(str(filepath), config)

                generated_images.append(str(filepath))
                batch_results.append({
                    "index": idx + 1,
                    "prompt": prompt,
                    "style": style,
                    "seed": seed,
                    "filepath": str(filepath),
                    "generation_time": gen_time
                })

            except Exception as e:
                batch_results.append({
                    "index": idx + 1,
                    "prompt": prompt,
                    "style": style,
                    "error": str(e)
                })

        # Save results
        with open(batch_dir / "results.json", 'w') as f:
            json.dump(batch_results, f, indent=2)

        completed = len([r for r in batch_results if "filepath" in r])
        failed = len([r for r in batch_results if "error" in r])

        final_status = f"Batch complete! {completed} images generated"
        if failed > 0:
            final_status += f", {failed} failed"
        final_status += f"\nOutput: {batch_dir}"

        yield generated_images, final_status, str(batch_dir), generated_images[-12:] if generated_images else []

    except Exception as e:
        yield [], f"Batch error: {str(e)}", "", []
    finally:
        batch_running = False


def stop_batch():
    """Stop the running batch"""
    global batch_stop_requested
    batch_stop_requested = True
    return "Stop requested... finishing current image"


def get_batch_gallery():
    """Get recent batch images"""
    batch_dir = OUTPUT_DIR / "batches"
    if not batch_dir.exists():
        return []

    # Get all images from recent batches
    images = []
    for batch in sorted(batch_dir.iterdir(), reverse=True)[:5]:
        if batch.is_dir():
            images.extend(sorted(batch.glob("*.png"))[:20])

    return [str(img) for img in images[:48]]


# ============================================================================
# BATCH CONFIG SAVE/LOAD
# ============================================================================

BATCH_CONFIGS_DIR = OUTPUT_DIR / "batch_configs"
BATCH_CONFIGS_DIR.mkdir(exist_ok=True)


def save_batch_config(
    batch_name: str,
    themes_text: str,
    variations_per_theme: int,
    styles_selected: List[str],
    images_per_combo: int,
    ollama_model: str,
    enhance_prompts: bool,
    aspect_ratio: str,
    quality_preset: str,
    random_seeds: bool
) -> str:
    """Save batch configuration to a JSON file"""
    if not batch_name.strip():
        return "Please enter a batch name"

    config = {
        "batch_name": batch_name,
        "themes": themes_text,
        "variations_per_theme": variations_per_theme,
        "styles": styles_selected,
        "images_per_combo": images_per_combo,
        "ollama_model": ollama_model,
        "enhance_prompts": enhance_prompts,
        "aspect_ratio": aspect_ratio,
        "quality_preset": quality_preset,
        "random_seeds": random_seeds,
        "saved_at": datetime.now().isoformat()
    }

    safe_name = "".join(c for c in batch_name[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = BATCH_CONFIGS_DIR / filename

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    return f"Saved: {filename}"


def get_saved_batch_configs() -> List[str]:
    """Get list of saved batch config files"""
    if not BATCH_CONFIGS_DIR.exists():
        return []
    configs = sorted(BATCH_CONFIGS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    return [f.name for f in configs]


def load_batch_config(config_name: str):
    """Load a saved batch configuration"""
    if not config_name:
        return [gr.update()] * 10 + ["No config selected"]

    filepath = BATCH_CONFIGS_DIR / config_name
    if not filepath.exists():
        return [gr.update()] * 10 + [f"Config not found: {config_name}"]

    try:
        with open(filepath, 'r') as f:
            config = json.load(f)

        return [
            gr.update(value=config.get('batch_name', '')),
            gr.update(value=config.get('themes', '')),
            gr.update(value=config.get('variations_per_theme', 3)),
            gr.update(value=config.get('styles', [])),
            gr.update(value=config.get('images_per_combo', 1)),
            gr.update(value=config.get('ollama_model', 'qwen2.5:7b-instruct')),
            gr.update(value=config.get('enhance_prompts', True)),
            gr.update(value=config.get('aspect_ratio', '1:1 (Square)')),
            gr.update(value=config.get('quality_preset', 'Standard')),
            gr.update(value=config.get('random_seeds', True)),
            f"Loaded: {config_name}\nThemes: {len(config.get('themes', '').split(chr(10)))} | Saved: {config.get('saved_at', 'unknown')[:10]}"
        ]
    except Exception as e:
        return [gr.update()] * 10 + [f"Error loading: {e}"]


# ============================================================================
# BATCH GALLERY BROWSER
# ============================================================================

def get_batch_directories() -> List[str]:
    """Get list of batch output directories"""
    batch_dir = OUTPUT_DIR / "batches"
    if not batch_dir.exists():
        return ["(No batches yet)"]

    dirs = sorted(
        [d.name for d in batch_dir.iterdir() if d.is_dir()],
        reverse=True
    )
    return dirs if dirs else ["(No batches yet)"]


def get_batch_images(batch_name: str, page: int = 0, per_page: int = 24) -> tuple:
    """Get images from a specific batch directory with pagination"""
    if not batch_name or batch_name == "(No batches yet)":
        return [], "No batch selected", 0, 0

    batch_path = OUTPUT_DIR / "batches" / batch_name
    if not batch_path.exists():
        return [], f"Batch not found: {batch_name}", 0, 0

    all_images = sorted(batch_path.glob("*.png"))
    total_images = len(all_images)
    total_pages = max(1, (total_images + per_page - 1) // per_page)

    page = max(0, min(page, total_pages - 1))
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, total_images)

    page_images = [str(img) for img in all_images[start_idx:end_idx]]

    info = f"Batch: {batch_name}\nImages: {start_idx + 1}-{end_idx} of {total_images} | Page {page + 1}/{total_pages}"

    return page_images, info, page, total_pages


def load_image_from_gallery(evt: gr.SelectData, current_batch: str):
    """Load config from a selected gallery image into the main UI"""
    if evt is None or evt.value is None:
        return [gr.update()] * 9 + ["No image selected"]

    # Get the image path from the event
    if isinstance(evt.value, dict):
        image_path = evt.value.get('image', {}).get('path', '')
    elif isinstance(evt.value, str):
        image_path = evt.value
    else:
        image_path = str(evt.value)

    if not image_path:
        return [gr.update()] * 9 + ["Could not get image path"]

    # Find the corresponding JSON config
    config_path = Path(image_path).with_suffix('.json')

    # If config not found at direct path (e.g., Gradio temp path),
    # try to find it in the actual batch directory
    if not config_path.exists() and current_batch and current_batch != "(No batches yet)":
        filename = Path(image_path).stem
        batch_dir = OUTPUT_DIR / "batches" / current_batch
        config_path = batch_dir / f"{filename}.json"

    if not config_path.exists():
        # Try to extract info from filename
        filename = Path(image_path).stem
        parts = filename.split('_')
        seed = None
        for p in parts:
            if p.startswith('s') and p[1:].isdigit():
                seed = int(p[1:])
                break

        return [
            gr.update(),  # prompt - can't recover
            gr.update(),  # style
            gr.update(),  # aspect_ratio
            gr.update(),  # quality
            gr.update(),  # steps
            gr.update(value=seed) if seed else gr.update(),  # seed
            gr.update(value=False) if seed else gr.update(),  # use_random
            gr.update(),  # use_ollama
            gr.update(),  # negative_prompt
            f"No config file found for this image.\nSeed extracted: {seed if seed else 'unknown'}"
        ]

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        return [
            gr.update(value=config.get('prompt', '')),
            gr.update(value=config.get('style', 'None')),
            gr.update(value=config.get('aspect_ratio', '1:1 (Square)')),
            gr.update(value=config.get('quality_preset', 'Standard')),
            gr.update(value=config.get('steps', 20)),
            gr.update(value=config.get('seed', 0)),
            gr.update(value=False),  # Uncheck random to use saved seed
            gr.update(value=config.get('use_ollama', False)),
            gr.update(value=config.get('negative_prompt', '')),
            f"Loaded from: {Path(image_path).name}\nPrompt: {config.get('prompt', '')[:80]}...\nSeed: {config.get('seed')}"
        ]
    except Exception as e:
        return [gr.update()] * 9 + [f"Error loading config: {e}"]


def refresh_batch_list():
    """Refresh the list of batch directories"""
    dirs = get_batch_directories()
    return gr.update(choices=dirs, value=dirs[0] if dirs else None)


def save_image_config(filepath: str, config: dict):
    """Save image configuration as a JSON sidecar file"""
    config_path = Path(filepath).with_suffix('.json')
    config['image_path'] = str(filepath)
    config['created_at'] = datetime.now().isoformat()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return str(config_path)


def load_image_config(config_path: str) -> dict:
    """Load image configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def load_config_to_ui(config_file):
    """Load a config file and return values for UI components"""
    if not config_file:
        return [gr.update()] * 8 + ["No file selected"]

    try:
        config = load_image_config(config_file.name if hasattr(config_file, 'name') else config_file)

        if "error" in config:
            return [gr.update()] * 8 + [f"Error: {config['error']}"]

        # Return updates for: prompt, style, aspect_ratio, quality, steps, seed, use_random, use_ollama, status
        return [
            gr.update(value=config.get('prompt', '')),
            gr.update(value=config.get('style', 'None')),
            gr.update(value=config.get('aspect_ratio', '1:1 (Square)')),
            gr.update(value=config.get('quality_preset', 'Standard')),
            gr.update(value=config.get('steps', 20)),
            gr.update(value=config.get('seed', 0)),
            gr.update(value=False),  # Uncheck random seed to use the saved seed
            gr.update(value=config.get('use_ollama', False)),
            f"Loaded config from: {config.get('image_path', 'unknown')}\nSeed: {config.get('seed')}"
        ]
    except Exception as e:
        return [gr.update()] * 8 + [f"Error loading config: {e}"]


def save_to_history(prompt: str, seed: int, image_size: str, steps: int,
                    filepath: str, generation_time: float, style: str):
    """Save generation details to history."""
    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text())
        except:
            history = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "seed": seed,
        "image_size": image_size,
        "steps": steps,
        "filepath": str(filepath),
        "generation_time": generation_time,
        "style": style,
    }

    history.insert(0, entry)
    history = history[:100]  # Keep last 100 entries

    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def generate_image(
    prompt: str,
    negative_prompt: str,
    style: str,
    aspect_ratio: str,
    custom_size: str,
    quality_preset: str,
    custom_steps: int,
    use_custom_steps: bool,
    seed: int,
    use_random_seed: bool,
    save_image: bool,
    use_ollama: bool = False,
    ollama_model: str = "qwen2.5:7b-instruct",
    batch_count: int = 1,
    ollama_length: str = "medium",
    ollama_complexity: str = "detailed",
    progress=gr.Progress()
):
    """Generate an image from the text prompt. Supports batch generation."""
    global model, model_loaded

    if not model_loaded:
        yield None, "Please load the model first! Click 'Load Model' above.", "", None
        return

    if not prompt.strip():
        yield None, "Please enter a prompt!", "", None
        return

    batch_count = int(batch_count)
    original_prompt = prompt.strip()

    try:
        # Determine image size (same for all batch items)
        if custom_size.strip():
            image_size = custom_size.strip()
        else:
            image_size = ASPECT_RATIOS.get(aspect_ratio, "auto")

        # Get inference steps (same for all batch items)
        if use_custom_steps:
            inference_steps = custom_steps
        else:
            inference_steps = QUALITY_PRESETS.get(quality_preset, {}).get("steps", 20)

        last_image = None
        last_seed = seed
        all_info = []

        for batch_idx in range(batch_count):
            batch_label = f"[{batch_idx + 1}/{batch_count}] " if batch_count > 1 else ""

            # Handle seed - generate new random seed for each batch item if random is enabled
            if use_random_seed:
                current_seed = random.randint(0, 2**32 - 1)
            else:
                # Use provided seed + offset for subsequent images in batch
                current_seed = (int(seed) + batch_idx) % (2**32)

            # Process wildcards if available - fresh random for each batch item
            processed_prompt = original_prompt
            wildcards_used = False

            if wildcard_available and wildcard_manager:
                if wildcard_manager.has_wildcards(original_prompt):
                    wildcards_used = True
                    progress((batch_idx * 0.9 / batch_count) + 0.01, desc=f"{batch_label}Processing wildcards...")
                    yield last_image, f"{batch_label}Processing wildcards...", "", current_seed
                    processed_prompt = wildcard_manager.process_prompt(original_prompt, seed=current_seed)

            # Apply style to prompt
            styled_prompt = apply_style(processed_prompt, style)

            # Enhance with Ollama if enabled (only on first batch item to save time)
            if use_ollama and ollama_available and batch_idx == 0:
                progress((batch_idx * 0.9 / batch_count) + 0.02, desc=f"{batch_label}Enhancing prompt with Ollama...")
                yield last_image, f"{batch_label}Enhancing prompt with {ollama_model} ({ollama_length}/{ollama_complexity})...", "", current_seed
                styled_prompt, enhance_status = enhance_with_ollama(
                    styled_prompt, ollama_model, style,
                    length=ollama_length, complexity=ollama_complexity
                )

            # Add negative prompt if provided
            full_prompt = styled_prompt
            if negative_prompt.strip():
                full_prompt = f"{styled_prompt}. Avoid: {negative_prompt.strip()}"

            progress((batch_idx * 0.9 / batch_count) + 0.05, desc=f"{batch_label}Initializing...")
            yield last_image, f"{batch_label}Starting generation...", "", current_seed

            start_time = time.time()

            progress((batch_idx * 0.9 / batch_count) + 0.1, desc=f"{batch_label}Generating with {inference_steps} steps...")
            yield last_image, f"{batch_label}Generating image ({inference_steps} steps)...", "", current_seed

            # Generate the image
            image = model.generate_image(
                prompt=full_prompt,
                seed=current_seed,
                image_size=image_size,
                diff_infer_steps=inference_steps,
                stream=True,
            )

            generation_time = time.time() - start_time

            # Generate filename and save if requested
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip()
            safe_prompt = safe_prompt.replace(" ", "_")
            filename = f"hunyuan_{timestamp}_{safe_prompt}_s{current_seed}.png"
            filepath = OUTPUT_DIR / filename

            if save_image:
                image.save(filepath)

                # Save config JSON sidecar for recreating this image
                config = {
                    "prompt": original_prompt,
                    "processed_prompt": processed_prompt if wildcards_used else None,
                    "styled_prompt": styled_prompt,
                    "style": style,
                    "negative_prompt": negative_prompt,
                    "seed": current_seed,
                    "image_size": image_size,
                    "aspect_ratio": aspect_ratio,
                    "steps": inference_steps,
                    "quality_preset": quality_preset,
                    "use_ollama": use_ollama,
                    "ollama_model": ollama_model if use_ollama else None,
                    "wildcards_used": wildcards_used,
                    "generation_time": generation_time,
                    "batch_index": batch_idx + 1 if batch_count > 1 else None,
                    "batch_total": batch_count if batch_count > 1 else None
                }
                config_path = save_image_config(str(filepath), config)

                save_status = f"Saved: {filename}"
                save_to_history(prompt, current_seed, image_size, inference_steps,
                              str(filepath), generation_time, style)
            else:
                save_status = "Not saved"

            # Build info string for this image
            if wildcards_used:
                item_info = f"[{batch_idx + 1}] {processed_prompt[:60]}... | Seed: {current_seed} | {generation_time:.1f}s"
            else:
                item_info = f"[{batch_idx + 1}] Seed: {current_seed} | {generation_time:.1f}s"
            all_info.append(item_info)

            last_image = image
            last_seed = current_seed

            # Show intermediate result
            progress_pct = ((batch_idx + 1) * 0.9 / batch_count)
            yield image, f"{batch_label}Generated! {save_status}", "\n".join(all_info), current_seed

        progress(1.0, desc="Done!")

        # Build final info string
        if batch_count > 1:
            total_info = f"Batch complete: {batch_count} images generated\n\n" + "\n".join(all_info)
            final_status = f"Batch complete! {batch_count} images generated"
        else:
            # Single image - show detailed info
            if wildcards_used:
                total_info = f"""Original: {original_prompt}
Processed: {processed_prompt}
Style: {style}
Size: {image_size}
Steps: {inference_steps}
Seed: {last_seed}
Time: {generation_time:.1f}s
{save_status if save_image else "Not saved (enable 'Auto-save' to save)"}"""
            else:
                total_info = f"""Prompt: {prompt}
Style: {style}
Size: {image_size}
Steps: {inference_steps}
Seed: {last_seed}
Time: {generation_time:.1f}s
{save_status if save_image else "Not saved (enable 'Auto-save' to save)"}"""
            final_status = "Generation complete!"

        yield last_image, final_status, total_info, last_seed

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        yield None, error_msg, "", seed


def regenerate_with_variation(prompt, negative_prompt, style, aspect_ratio,
                              custom_size, quality_preset, custom_steps,
                              use_custom_steps, last_seed, save_image, variation_amount):
    """Regenerate with a slight variation of the seed."""
    if last_seed is None or last_seed == 0:
        new_seed = random.randint(0, 2**32 - 1)
    else:
        # Create a variation by adding a small random offset
        offset = random.randint(1, max(1, int(variation_amount)))
        new_seed = (int(last_seed) + offset) % (2**32)

    yield from generate_image(
        prompt, negative_prompt, style, aspect_ratio, custom_size,
        quality_preset, custom_steps, use_custom_steps,
        new_seed, False, save_image
    )


def get_gallery_images():
    """Get list of recently generated images."""
    images = sorted(OUTPUT_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)[:24]
    return [str(img) for img in images]


def refresh_gallery():
    """Refresh the gallery with latest images."""
    return get_gallery_images()


def load_config_from_main_gallery(evt: gr.SelectData):
    """Load config from a selected image in the main gallery into the UI"""
    if evt is None or evt.value is None:
        return [gr.update()] * 9 + ["No image selected"]

    # Get the image path from the event
    if isinstance(evt.value, dict):
        image_path = evt.value.get('image', {}).get('path', '')
    elif isinstance(evt.value, str):
        image_path = evt.value
    else:
        image_path = str(evt.value)

    if not image_path:
        return [gr.update()] * 9 + ["Could not get image path"]

    # Find the corresponding JSON config
    config_path = Path(image_path).with_suffix('.json')

    # If config not found at direct path (e.g., Gradio temp path),
    # try to find it in the main outputs directory
    if not config_path.exists():
        filename = Path(image_path).stem
        config_path = OUTPUT_DIR / f"{filename}.json"

    if not config_path.exists():
        # Try to extract info from filename
        filename = Path(image_path).stem
        parts = filename.split('_')
        seed = None
        for p in parts:
            if p.startswith('s') and p[1:].isdigit():
                seed = int(p[1:])
                break

        return [
            gr.update(),  # prompt
            gr.update(),  # style
            gr.update(),  # aspect_ratio
            gr.update(),  # quality
            gr.update(),  # steps
            gr.update(value=seed) if seed else gr.update(),  # seed
            gr.update(value=False) if seed else gr.update(),  # use_random
            gr.update(),  # use_ollama
            gr.update(),  # negative_prompt
            f"No config file found. Seed extracted: {seed if seed else 'unknown'}"
        ]

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        return [
            gr.update(value=config.get('prompt', '')),
            gr.update(value=config.get('style', 'None')),
            gr.update(value=config.get('aspect_ratio', '1:1 (Square)')),
            gr.update(value=config.get('quality_preset', 'Standard')),
            gr.update(value=config.get('steps', 20)),
            gr.update(value=config.get('seed', 0)),
            gr.update(value=False),  # Uncheck random to use saved seed
            gr.update(value=config.get('use_ollama', False)),
            gr.update(value=config.get('negative_prompt', '')),
            f"Loaded: {Path(image_path).name}\nPrompt: {config.get('prompt', '')[:80]}...\nSeed: {config.get('seed')}"
        ]
    except Exception as e:
        return [gr.update()] * 9 + [f"Error loading config: {e}"]


def clear_outputs():
    """Clear the output display."""
    return None, "", ""


def update_steps_from_preset(preset):
    """Update the custom steps slider when preset changes."""
    steps = QUALITY_PRESETS.get(preset, {}).get("steps", 20)
    return steps


def copy_seed(seed):
    """Return the seed for copying."""
    return int(seed) if seed else 0


# Example prompts for inspiration
EXAMPLE_PROMPTS = [
    ("Majestic Lion", "A majestic lion with a flowing golden mane, standing on a rocky outcrop at sunset, dramatic lighting"),
    ("Cozy Coffee Shop", "A cozy coffee shop interior with warm lighting, wooden furniture, plants hanging from the ceiling"),
    ("Cyberpunk City", "A futuristic cityscape at night with neon lights reflecting on wet streets, cyberpunk aesthetic"),
    ("Japanese Garden", "A serene Japanese garden with a koi pond, cherry blossoms falling, morning mist"),
    ("Cute Robot", "A cute robot holding a bouquet of flowers, pastel colors, soft lighting, kawaii style"),
    ("Astronaut", "An astronaut floating in space with Earth in the background, cinematic lighting"),
    ("Medieval Castle", "A medieval castle on a cliff overlooking the sea, stormy sky, dramatic atmosphere"),
    ("Gourmet Burger", "A delicious gourmet burger with melting cheese, fresh vegetables, studio food photography"),
    ("Portrait Woman", "Portrait of a young woman with flowing red hair, green eyes, soft natural lighting"),
    ("Mountain Lake", "A pristine mountain lake reflecting snow-capped peaks, golden hour, landscape photography"),
    ("Steampunk Airship", "A magnificent steampunk airship flying through clouds at sunset, intricate brass details"),
    ("Abstract Art", "Abstract flowing shapes in vibrant colors, modern art, geometric patterns intertwining"),
]


# Build the Gradio interface
def create_ui():
    """Create and configure the Gradio interface."""

    with gr.Blocks(
        title="HunyuanImage-3.0 Generator",
    ) as app:

        # Header
        gr.Markdown("""
        # HunyuanImage-3.0 Image Generator
        **Local AI Image Generation** - Powered by Tencent's 80B MoE Model (Quantized)

        Generate high-quality images from text descriptions. Runs entirely on your local GPU.
        """)

        with gr.Row():
            # Left column - Input controls
            with gr.Column(scale=1):

                # Model status section
                with gr.Group():
                    gr.Markdown("### Model Status")
                    model_status = gr.Textbox(
                        value="Model loaded on RTX PRO 6000 Blackwell (96GB) - Ready!",
                        interactive=False,
                        show_label=False,
                    )

                # Prompt section
                gr.Markdown("### Prompt")
                prompt = gr.Textbox(
                    label="Describe your image",
                    placeholder="A stunning landscape with mountains, a crystal clear lake reflecting the sky...",
                    lines=3,
                    max_lines=8,
                )

                with gr.Row():
                    style = gr.Dropdown(
                        label="Style Preset",
                        choices=list(STYLE_PRESETS.keys()),
                        value="None",
                        scale=2
                    )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt (what to avoid)",
                    placeholder="blurry, low quality, distorted...",
                    lines=1,
                )

                # Wildcards section
                with gr.Accordion("Wildcards (Dynamic Prompt Variables)", open=False):
                    if wildcard_available and wildcard_manager:
                        gr.Markdown("""
                        **Insert `[wildcard]` tags to generate random variations!**
                        Example: `A [animal] in a [landscape]` → `A tiger in a forest`
                        """)
                        with gr.Row():
                            wildcard_category = gr.Dropdown(
                                label="Category",
                                choices=["all"] + sorted(set(
                                    k.split('-')[0] for k in wildcard_manager.get_available_wildcards()
                                )),
                                value="all",
                                scale=1
                            )
                            wildcard_dropdown = gr.Dropdown(
                                label="Select Wildcard to Insert",
                                choices=wildcard_manager.get_available_wildcards(),
                                interactive=True,
                                scale=3
                            )
                            wildcard_insert_btn = gr.Button("Insert", size="sm", scale=1)

                        wildcard_preview = gr.Textbox(
                            label="Preview (sample values)",
                            interactive=False,
                            lines=2
                        )

                        with gr.Row():
                            wildcard_search = gr.Textbox(
                                label="Search wildcards",
                                placeholder="Type to filter...",
                                scale=3
                            )
                            wildcard_count = gr.Markdown(
                                f"**{len(wildcard_manager.get_available_wildcards())}** wildcards available"
                            )
                    else:
                        gr.Markdown("*Wildcards not available. Check wildcards.json file.*")

                # Example prompts
                with gr.Accordion("Example Prompts", open=False):
                    with gr.Row():
                        for i in range(0, 6, 2):
                            with gr.Column():
                                for j in range(2):
                                    if i + j < len(EXAMPLE_PROMPTS):
                                        name, text = EXAMPLE_PROMPTS[i + j]
                                        btn = gr.Button(name, size="sm")
                                        btn.click(fn=lambda t=text: t, outputs=prompt)
                    with gr.Row():
                        for i in range(6, 12, 2):
                            with gr.Column():
                                for j in range(2):
                                    if i + j < len(EXAMPLE_PROMPTS):
                                        name, text = EXAMPLE_PROMPTS[i + j]
                                        btn = gr.Button(name, size="sm")
                                        btn.click(fn=lambda t=text: t, outputs=prompt)

                # Settings in tabs
                with gr.Tabs():
                    with gr.Tab("Size"):
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="1:1 (Square)",
                        )
                        custom_size = gr.Textbox(
                            label="Custom Size (optional)",
                            placeholder="e.g., 1024x768",
                            info="Overrides aspect ratio if set"
                        )

                    with gr.Tab("Quality"):
                        quality_preset = gr.Radio(
                            label="Quality Preset",
                            choices=list(QUALITY_PRESETS.keys()),
                            value="Standard",
                        )
                        gr.Markdown("*Or set custom steps:*")
                        with gr.Row():
                            use_custom_steps = gr.Checkbox(
                                label="Use custom",
                                value=False,
                                scale=1
                            )
                            custom_steps = gr.Slider(
                                label="Steps",
                                minimum=10,
                                maximum=50,
                                value=20,
                                step=1,
                                scale=3
                            )

                    with gr.Tab("Seed"):
                        with gr.Row():
                            seed = gr.Number(
                                label="Seed",
                                value=0,
                                precision=0,
                                scale=2
                            )
                            use_random_seed = gr.Checkbox(
                                label="Random",
                                value=True,
                                scale=1
                            )
                        gr.Markdown("*Same seed + same prompt = same image*")

                    with gr.Tab("Ollama"):
                        with gr.Group():
                            gr.Markdown("**Server Control**")
                            ollama_status = gr.Textbox(
                                label="Status",
                                value=check_ollama_status() if ollama_available else "Ollama modules not installed",
                                interactive=False
                            )
                            with gr.Row():
                                start_ollama_btn = gr.Button("Start", size="sm", scale=1)
                                stop_ollama_btn = gr.Button("Stop", size="sm", scale=1)
                                refresh_ollama = gr.Button("Refresh", size="sm", scale=1)

                        with gr.Group():
                            gr.Markdown("**Prompt Enhancement**")
                            use_ollama = gr.Checkbox(
                                label="Enhance prompts with Ollama",
                                value=False,
                                info="Use local LLM to enhance your prompt before generation"
                            )
                            initial_models = get_ollama_models_list() if ollama_available else OLLAMA_MODELS
                            ollama_model = gr.Dropdown(
                                label="Model",
                                choices=initial_models,
                                value=initial_models[0] if initial_models else "qwen2.5:7b-instruct",
                                info="Smaller = faster, Larger = more creative"
                            )

                        with gr.Group():
                            gr.Markdown("**Prompt Length & Complexity**")
                            with gr.Row():
                                ollama_length = gr.Dropdown(
                                    label="Length",
                                    choices=["minimal", "short", "medium", "long", "detailed"],
                                    value="medium",
                                    info="How many words in the enhanced prompt"
                                )
                                ollama_complexity = gr.Dropdown(
                                    label="Complexity",
                                    choices=["simple", "basic", "moderate", "detailed", "complex"],
                                    value="detailed",
                                    info="How much detail to add"
                                )
                            gr.Markdown("""
                            *Length:* minimal (15-30 words) → detailed (150-250 words)
                            *Complexity:* simple (subject only) → complex (full cinematic detail)
                            """)

                        with gr.Accordion("Install/Remove Models", open=False):
                            new_model_name = gr.Textbox(
                                label="Model to Install",
                                placeholder="e.g., llama3.2:3b, mistral:7b, gemma2:9b",
                                info="Enter model name from ollama.com/library"
                            )
                            with gr.Row():
                                pull_model_btn = gr.Button("Install Model", size="sm", variant="primary")
                                delete_model_btn = gr.Button("Delete Selected", size="sm", variant="stop")
                            model_action_status = gr.Textbox(label="Status", interactive=False)
                            gr.Markdown("""
                            **Popular models:** `llama3.2:3b` (2GB), `mistral:7b` (4GB), `gemma2:9b` (5GB), `qwen2.5:14b` (9GB)
                            """)

                # Save option and Load config
                save_image = gr.Checkbox(
                    label="Auto-save images (with JSON config for recreating)",
                    value=True,
                )

                with gr.Accordion("Load Saved Config", open=False):
                    config_file = gr.File(
                        label="Select .json config file",
                        file_types=[".json"],
                        type="filepath"
                    )
                    load_config_btn = gr.Button("Load Config", size="sm")
                    load_config_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )
                    gr.Markdown("*Each saved image has a .json file with the same name. Load it to recreate the image.*")

                # Generate buttons
                gr.Markdown("### Generate")
                with gr.Row():
                    batch_count = gr.Slider(
                        label="Batch Count (images to generate)",
                        minimum=1,
                        maximum=20,
                        value=1,
                        step=1,
                        info="Generate multiple images - wildcards get new random values each time"
                    )
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate Image",
                        variant="primary",
                        size="lg",
                        scale=3
                    )
                with gr.Row():
                    variation_btn = gr.Button(
                        "Create Variation",
                        size="sm",
                        scale=2
                    )
                    variation_amount = gr.Slider(
                        label="Variation",
                        minimum=1,
                        maximum=1000,
                        value=100,
                        step=1,
                        scale=1
                    )

            # Right column - Output display
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=520,
                )

                with gr.Row():
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=3
                    )
                    last_seed = gr.Number(
                        label="Seed Used",
                        interactive=False,
                        scale=1
                    )

                info_text = gr.Textbox(
                    label="Generation Details",
                    interactive=False,
                    lines=7,
                )

                with gr.Row():
                    copy_seed_btn = gr.Button("Copy Seed", size="sm")
                    clear_btn = gr.Button("Clear", size="sm")

        # Gallery section
        with gr.Accordion("Recent Generations", open=False):
            with gr.Row():
                refresh_btn = gr.Button("Refresh Gallery", size="sm")
                gr.Markdown(f"*Images saved to: `{OUTPUT_DIR}`*")
            gallery = gr.Gallery(
                label="Recent Images",
                columns=6,
                rows=2,
                height=250,
                value=get_gallery_images,
            )

        # Prompt Generator section (Ollama-powered)
        with gr.Accordion("Prompt Generator (Ollama)", open=False):
            gr.Markdown("**Generate creative prompts using local LLM**")
            with gr.Row():
                with gr.Column(scale=2):
                    gen_theme = gr.Textbox(
                        label="Theme or Concept",
                        placeholder="e.g., 'cyberpunk cities', 'underwater creatures', 'fantasy landscapes'",
                    )
                    with gr.Row():
                        gen_count = gr.Slider(
                            label="Number of prompts",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1
                        )
                        gen_style = gr.Dropdown(
                            label="Style",
                            choices=list(STYLE_PRESETS.keys()),
                            value="None"
                        )
                    gen_model = gr.Dropdown(
                        label="Ollama Model",
                        choices=OLLAMA_MODELS,
                        value="qwen2.5:7b-instruct"
                    )
                    gen_btn = gr.Button("Generate Prompts", variant="primary")
                with gr.Column(scale=3):
                    gen_output = gr.Textbox(
                        label="Generated Prompts",
                        lines=12,
                        max_lines=20,
                        interactive=True,
                        placeholder="Generated prompts will appear here..."
                    )
                    with gr.Row():
                        use_first_btn = gr.Button("Use First Prompt", size="sm")
                        gr.Markdown("*Click a generated prompt to copy it above*")

        # Batch Generation Tab
        with gr.Accordion("Batch Generation (Mass Image Creation)", open=False):
            gr.Markdown("""
            **Generate hundreds of images with mixed themes, styles, and variations**

            Enter themes (one per line), set variations and styles, and let Ollama + HunyuanImage create a massive batch.
            """)

            with gr.Tabs():
                # Tab 1: Batch Configuration
                with gr.Tab("Configure Batch"):
                    with gr.Row():
                        # Left side - Configuration
                        with gr.Column(scale=1):
                            batch_name = gr.Textbox(
                                label="Batch Name",
                                value="my_batch",
                                placeholder="Name for this batch run"
                            )

                            batch_themes = gr.Textbox(
                                label="Themes/Prompts (one per line)",
                                placeholder="cyberpunk city at night\nunderwater ancient temple\nfantasy forest with magical creatures\nfuturistic space station",
                                lines=8,
                                max_lines=20
                            )

                            with gr.Row():
                                batch_variations = gr.Slider(
                                    label="Variations per theme",
                                    minimum=1,
                                    maximum=20,
                                    value=3,
                                    step=1,
                                    info="Ollama generates this many variations of each theme"
                                )
                                batch_images_per = gr.Slider(
                                    label="Images per combo",
                                    minimum=1,
                                    maximum=10,
                                    value=1,
                                    step=1,
                                    info="How many images per prompt+style combination"
                                )

                            batch_styles = gr.CheckboxGroup(
                                label="Styles to apply (select multiple)",
                                choices=list(STYLE_PRESETS.keys()),
                                value=["Photorealistic", "Cinematic", "Digital Art"],
                                info="Each prompt variation will be generated in all selected styles"
                            )

                            with gr.Row():
                                batch_aspect = gr.Dropdown(
                                    label="Aspect Ratio",
                                    choices=list(ASPECT_RATIOS.keys()),
                                    value="1:1 (Square)"
                                )
                                batch_quality = gr.Dropdown(
                                    label="Quality",
                                    choices=list(QUALITY_PRESETS.keys()),
                                    value="Standard"
                                )

                            with gr.Row():
                                batch_ollama_model = gr.Dropdown(
                                    label="Ollama Model",
                                    choices=get_ollama_models_list() if ollama_available else OLLAMA_MODELS,
                                    value="qwen2.5:7b-instruct"
                                )
                                batch_enhance = gr.Checkbox(
                                    label="Enhance prompts",
                                    value=True,
                                    info="Use Ollama to enhance each prompt"
                                )

                            with gr.Row():
                                batch_ollama_length = gr.Dropdown(
                                    label="Prompt Length",
                                    choices=["minimal", "short", "medium", "long", "detailed"],
                                    value="medium",
                                    info="How long the generated prompts should be"
                                )
                                batch_ollama_complexity = gr.Dropdown(
                                    label="Prompt Complexity",
                                    choices=["simple", "basic", "moderate", "detailed", "complex"],
                                    value="detailed",
                                    info="How much detail to include"
                                )

                            batch_random_seeds = gr.Checkbox(
                                label="Random seeds for each image",
                                value=True
                            )

                        # Right side - Preview, controls, and save/load
                        with gr.Column(scale=1):
                            batch_preview = gr.Markdown(
                                value="Enter themes and settings to see batch preview..."
                            )

                            with gr.Row():
                                batch_calculate_btn = gr.Button("Calculate", variant="secondary", size="sm")
                                batch_start_btn = gr.Button("Start Batch", variant="primary", size="lg")
                                batch_stop_btn = gr.Button("Stop", variant="stop", size="sm")

                            batch_status = gr.Textbox(
                                label="Status",
                                value="Ready",
                                interactive=False,
                                lines=3
                            )

                            batch_output_dir = gr.Textbox(
                                label="Output Directory",
                                interactive=False
                            )

                            # Save/Load batch configs
                            gr.Markdown("---\n**Save/Load Batch Configuration**")
                            with gr.Row():
                                batch_save_btn = gr.Button("Save Config", size="sm")
                                batch_config_dropdown = gr.Dropdown(
                                    label="Saved Configs",
                                    choices=get_saved_batch_configs(),
                                    value=None,
                                    allow_custom_value=False
                                )
                                batch_load_btn = gr.Button("Load", size="sm")
                                batch_refresh_configs_btn = gr.Button("↻", size="sm", scale=0)

                            batch_config_status = gr.Textbox(
                                label="Config Status",
                                interactive=False,
                                lines=2
                            )

                    # Gallery showing current batch progress
                    gr.Markdown("---\n**Current Batch Progress**")
                    batch_gallery = gr.Gallery(
                        label="Generated Images (live update)",
                        columns=6,
                        rows=2,
                        height=200,
                        object_fit="cover"
                    )

                # Tab 2: Browse Batches (Gallery)
                with gr.Tab("Browse Batches"):
                    gr.Markdown("**Browse generated batches and click any image to load it into the main generator**")

                    with gr.Row():
                        browse_batch_dropdown = gr.Dropdown(
                            label="Select Batch",
                            choices=get_batch_directories(),
                            value=get_batch_directories()[0] if get_batch_directories() else None,
                            scale=3
                        )
                        browse_refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)

                    browse_info = gr.Textbox(
                        label="Batch Info",
                        interactive=False,
                        lines=2
                    )

                    browse_gallery = gr.Gallery(
                        label="Click an image to load its settings",
                        columns=6,
                        rows=4,
                        height=400,
                        object_fit="cover",
                        allow_preview=True
                    )

                    with gr.Row():
                        browse_page_state = gr.State(value=0)
                        browse_total_pages = gr.State(value=1)
                        browse_prev_btn = gr.Button("← Previous", size="sm")
                        browse_page_info = gr.Markdown("Page 1 of 1")
                        browse_next_btn = gr.Button("Next →", size="sm")

                    browse_load_status = gr.Textbox(
                        label="Load Status",
                        interactive=False,
                        lines=3,
                        placeholder="Click an image above to load its configuration into the main generator..."
                    )

                # Tab 3: Example Ideas
                with gr.Tab("Example Ideas"):
                    gr.Markdown("""
                    **Quick Examples - Copy these themes:**

                    **Sci-Fi Collection (5 themes x 3 variations x 3 styles = 45 images):**
                    ```
                    alien planet landscape with two moons
                    cyberpunk street market with neon signs
                    space station orbiting a gas giant
                    android in a futuristic city
                    abandoned spaceship interior
                    ```

                    **Fantasy Collection:**
                    ```
                    dragon flying over a medieval castle
                    enchanted forest with glowing mushrooms
                    wizard tower on a floating island
                    underwater mermaid kingdom
                    magical library with floating books
                    ```

                    **Portrait Collection:**
                    ```
                    portrait of an elderly wise man
                    young woman with flowers in hair
                    warrior in ornate armor
                    steampunk inventor with goggles
                    ethereal fairy queen
                    ```

                    **Nature & Landscapes:**
                    ```
                    misty mountain peak at sunrise
                    tropical waterfall in dense jungle
                    northern lights over frozen lake
                    desert oasis with palm trees
                    cherry blossom garden in spring
                    ```

                    **Architecture & Interiors:**
                    ```
                    ancient greek temple ruins
                    modern minimalist living room
                    gothic cathedral interior
                    japanese zen garden with temple
                    art deco luxury hotel lobby
                    ```
                    """)

        # Footer
        gr.Markdown("""
        ---
        **Quick Tips:**
        - Be descriptive for better results
        - Use style presets to quickly change the artistic direction
        - Save the seed to recreate or create variations of images you like
        - Enable "Ollama" tab to enhance prompts with local LLM
        - Use "Batch Generation" to create hundreds of images with mixed themes, styles, and variations

        **Links:** [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0) |
        [Quantized Model](https://huggingface.co/Disty0/HunyuanImage3-SDNQ-uint4-svd-r32)
        """)

        # Event handlers
        quality_preset.change(
            fn=update_steps_from_preset,
            inputs=[quality_preset],
            outputs=[custom_steps]
        )

        # Ollama server control handlers
        refresh_ollama.click(
            fn=check_ollama_status,
            outputs=[ollama_status]
        )

        start_ollama_btn.click(
            fn=start_ollama_server,
            outputs=[ollama_status, ollama_model]
        )

        stop_ollama_btn.click(
            fn=stop_ollama_server,
            outputs=[ollama_status]
        )

        pull_model_btn.click(
            fn=pull_ollama_model,
            inputs=[new_model_name],
            outputs=[model_action_status, ollama_model]
        )

        delete_model_btn.click(
            fn=delete_ollama_model,
            inputs=[ollama_model],
            outputs=[model_action_status, ollama_model]
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                style,
                aspect_ratio,
                custom_size,
                quality_preset,
                custom_steps,
                use_custom_steps,
                seed,
                use_random_seed,
                save_image,
                use_ollama,
                ollama_model,
                batch_count,
                ollama_length,
                ollama_complexity,
            ],
            outputs=[output_image, status_text, info_text, last_seed],
        )

        variation_btn.click(
            fn=regenerate_with_variation,
            inputs=[
                prompt,
                negative_prompt,
                style,
                aspect_ratio,
                custom_size,
                quality_preset,
                custom_steps,
                use_custom_steps,
                last_seed,
                save_image,
                variation_amount,
            ],
            outputs=[output_image, status_text, info_text, last_seed],
        )

        clear_btn.click(
            fn=clear_outputs,
            outputs=[output_image, status_text, info_text],
        )

        copy_seed_btn.click(
            fn=copy_seed,
            inputs=[last_seed],
            outputs=[seed],
        )

        refresh_btn.click(
            fn=refresh_gallery,
            outputs=gallery,
        )

        # Main gallery click-to-load handler
        gallery.select(
            fn=load_config_from_main_gallery,
            inputs=[],
            outputs=[
                prompt,
                style,
                aspect_ratio,
                quality_preset,
                custom_steps,
                seed,
                use_random_seed,
                use_ollama,
                negative_prompt,
                info_text
            ]
        )

        # Load config handler
        load_config_btn.click(
            fn=load_config_to_ui,
            inputs=[config_file],
            outputs=[
                prompt,
                style,
                aspect_ratio,
                quality_preset,
                custom_steps,
                seed,
                use_random_seed,
                use_ollama,
                load_config_status
            ]
        )

        # Prompt generator handlers
        def generate_and_format(theme, count, model, style):
            if not theme.strip():
                return "Please enter a theme or concept"
            prompts = generate_prompts_ollama(theme, int(count), model, style)
            return "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(prompts))

        def get_first_prompt(text):
            if not text or "[1]" not in text:
                return ""
            lines = text.split("\n\n")
            if lines:
                first = lines[0]
                if first.startswith("[1] "):
                    return first[4:]
            return ""

        gen_btn.click(
            fn=generate_and_format,
            inputs=[gen_theme, gen_count, gen_model, gen_style],
            outputs=[gen_output]
        )

        use_first_btn.click(
            fn=get_first_prompt,
            inputs=[gen_output],
            outputs=[prompt]
        )

        # Allow Enter key to generate
        prompt.submit(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                style,
                aspect_ratio,
                custom_size,
                quality_preset,
                custom_steps,
                use_custom_steps,
                seed,
                use_random_seed,
                save_image,
                use_ollama,
                ollama_model,
                batch_count,
                ollama_length,
                ollama_complexity,
            ],
            outputs=[output_image, status_text, info_text, last_seed],
        )

        # Batch generation handlers
        batch_calculate_btn.click(
            fn=calculate_batch_total,
            inputs=[batch_themes, batch_variations, batch_styles, batch_images_per],
            outputs=[batch_preview]
        )

        batch_start_btn.click(
            fn=run_batch_generation,
            inputs=[
                batch_themes,
                batch_variations,
                batch_styles,
                batch_images_per,
                batch_ollama_model,
                batch_enhance,
                batch_aspect,
                batch_quality,
                batch_random_seeds,
                batch_name,
                batch_ollama_length,
                batch_ollama_complexity,
            ],
            outputs=[gr.State(), batch_status, batch_output_dir, batch_gallery]
        )

        batch_stop_btn.click(
            fn=stop_batch,
            outputs=[batch_status]
        )

        # Auto-calculate batch on setting changes
        for component in [batch_themes, batch_variations, batch_styles, batch_images_per]:
            component.change(
                fn=calculate_batch_total,
                inputs=[batch_themes, batch_variations, batch_styles, batch_images_per],
                outputs=[batch_preview]
            )

        # ============================================================
        # BATCH CONFIG SAVE/LOAD HANDLERS
        # ============================================================

        batch_save_btn.click(
            fn=save_batch_config,
            inputs=[
                batch_name,
                batch_themes,
                batch_variations,
                batch_styles,
                batch_images_per,
                batch_ollama_model,
                batch_enhance,
                batch_aspect,
                batch_quality,
                batch_random_seeds
            ],
            outputs=[batch_config_status]
        )

        batch_load_btn.click(
            fn=load_batch_config,
            inputs=[batch_config_dropdown],
            outputs=[
                batch_name,
                batch_themes,
                batch_variations,
                batch_styles,
                batch_images_per,
                batch_ollama_model,
                batch_enhance,
                batch_aspect,
                batch_quality,
                batch_random_seeds,
                batch_config_status
            ]
        )

        batch_refresh_configs_btn.click(
            fn=lambda: gr.update(choices=get_saved_batch_configs()),
            outputs=[batch_config_dropdown]
        )

        # ============================================================
        # BATCH BROWSER HANDLERS
        # ============================================================

        def update_batch_browser(batch_name, page=0):
            """Update the batch browser gallery and info"""
            images, info, current_page, total_pages = get_batch_images(batch_name, page)
            page_text = f"Page {current_page + 1} of {total_pages}"
            return images, info, current_page, total_pages, page_text

        def browse_prev_page(batch_name, current_page):
            """Go to previous page"""
            new_page = max(0, current_page - 1)
            return update_batch_browser(batch_name, new_page)

        def browse_next_page(batch_name, current_page, total_pages):
            """Go to next page"""
            new_page = min(total_pages - 1, current_page + 1)
            return update_batch_browser(batch_name, new_page)

        browse_batch_dropdown.change(
            fn=lambda batch_name: update_batch_browser(batch_name, 0),
            inputs=[browse_batch_dropdown],
            outputs=[browse_gallery, browse_info, browse_page_state, browse_total_pages, browse_page_info]
        )

        browse_refresh_btn.click(
            fn=refresh_batch_list,
            outputs=[browse_batch_dropdown]
        )

        browse_prev_btn.click(
            fn=browse_prev_page,
            inputs=[browse_batch_dropdown, browse_page_state],
            outputs=[browse_gallery, browse_info, browse_page_state, browse_total_pages, browse_page_info]
        )

        browse_next_btn.click(
            fn=browse_next_page,
            inputs=[browse_batch_dropdown, browse_page_state, browse_total_pages],
            outputs=[browse_gallery, browse_info, browse_page_state, browse_total_pages, browse_page_info]
        )

        # Click on image in browse gallery to load its config
        browse_gallery.select(
            fn=load_image_from_gallery,
            inputs=[browse_batch_dropdown],
            outputs=[
                prompt,
                style,
                aspect_ratio,
                quality_preset,
                custom_steps,
                seed,
                use_random_seed,
                use_ollama,
                negative_prompt,
                browse_load_status
            ]
        )

        # ============================================================
        # WILDCARD HANDLERS
        # ============================================================

        if wildcard_available and wildcard_manager:
            def filter_wildcards_by_category(category):
                """Filter wildcards by category prefix"""
                all_wildcards = wildcard_manager.get_available_wildcards()
                if category == "all":
                    return gr.update(choices=all_wildcards)
                filtered = [w for w in all_wildcards if w.startswith(category + "-") or w == category]
                if not filtered:
                    filtered = [w for w in all_wildcards if category in w]
                return gr.update(choices=filtered if filtered else all_wildcards)

            def search_wildcards(query):
                """Search wildcards by name"""
                if not query.strip():
                    return gr.update(choices=wildcard_manager.get_available_wildcards())
                results = wildcard_manager.search_wildcards(query)
                return gr.update(choices=results if results else wildcard_manager.get_available_wildcards())

            def show_wildcard_preview(wildcard_name):
                """Show preview of wildcard values"""
                if not wildcard_name:
                    return ""
                return preview_wildcard(wildcard_name)

            def insert_wildcard_to_prompt(current_prompt, wildcard_name):
                """Insert wildcard tag into prompt"""
                if not wildcard_name:
                    return current_prompt
                return insert_wildcard(current_prompt, wildcard_name)

            wildcard_category.change(
                fn=filter_wildcards_by_category,
                inputs=[wildcard_category],
                outputs=[wildcard_dropdown]
            )

            wildcard_search.change(
                fn=search_wildcards,
                inputs=[wildcard_search],
                outputs=[wildcard_dropdown]
            )

            wildcard_dropdown.change(
                fn=show_wildcard_preview,
                inputs=[wildcard_dropdown],
                outputs=[wildcard_preview]
            )

            wildcard_insert_btn.click(
                fn=insert_wildcard_to_prompt,
                inputs=[prompt, wildcard_dropdown],
                outputs=[prompt]
            )

            # Also insert on dropdown select (double-click behavior)
            wildcard_dropdown.select(
                fn=insert_wildcard_to_prompt,
                inputs=[prompt, wildcard_dropdown],
                outputs=[prompt]
            )

    return app


def load_model_on_startup():
    """Load the model at startup."""
    global model, model_loaded

    if model_loaded:
        return

    try:
        from transformers import AutoModelForCausalLM
        from sdnq import SDNQConfig  # Registers SDNQ into transformers

        model_id = "/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage3-SDNQ"

        print("Loading quantized HunyuanImage-3.0 model on GPU 0...")
        print("This may take 1-2 minutes...")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            moe_impl="eager",
        )
        model.load_tokenizer(model_id)
        model_loaded = True

        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    print("=" * 50)
    print("HunyuanImage-3.0 Image Generator")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Set CUDA device - Use GPU 1 (RTX PRO 6000 Blackwell 96GB)
    # Note: nvidia-smi shows Blackwell as GPU 0, but CUDA/PyTorch sees it as GPU 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Load model before starting UI
    load_model_on_startup()

    print()
    print("Starting web interface...")

    # Create and launch the UI
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
