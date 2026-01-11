#!/usr/bin/env python3
"""
HunyuanImage-3.0 Quantized Model UI - Version 2 (Modular)
New layout with GPU selector, Ollama under prompt, batch-first design.

Usage:
    source hunyuan_env/bin/activate
    python hunyuan_ui_v2.py

Runs on port 7860 (same as original).
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import random
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import gradio as gr
from PIL import Image

# Import modular UI components
from ui.state import get_state, init_gpus, set_gpu
from ui.constants import (
    OUTPUT_DIR, MODEL_PATH, ASPECT_RATIOS, QUALITY_PRESETS,
    DEFAULT_STYLE_PRESETS, STYLE_PRESETS_FILE
)
from ui.components import (
    create_system_bar, wire_system_bar_events,
    create_prompt_input, wire_prompt_input_events,
    create_gen_settings, wire_gen_settings_events,
    create_output_display, wire_output_display_events,
    create_batch_panel,
    get_steps_from_quality, get_size_from_aspect,
)
from core.model_manager import load_model, unload_model, get_model_status, is_model_loaded, get_model

# Initialize integrations
state = get_state()

# Ollama
try:
    from ollama_prompts import PromptEnhancer, OllamaClient
    from prompt_generator import PromptGenerator
    from ollama_manager import OllamaManager
    state.ollama_manager = OllamaManager()
    state.ollama_available = True
    print("[INIT] Ollama available")
except ImportError:
    print("[INIT] Ollama not available")

# Wildcards
try:
    from wildcard_utils import WildcardManager
    state.wildcard_manager = WildcardManager(
        json_path=Path(__file__).parent / "wildcards.json"
    )
    state.wildcard_available = True
    print("[INIT] Wildcards available")
except ImportError:
    print("[INIT] Wildcards not available")

# Load style presets
def load_style_presets() -> dict:
    if STYLE_PRESETS_FILE.exists():
        try:
            with open(STYLE_PRESETS_FILE, 'r') as f:
                presets = json.load(f)
            if "None" not in presets:
                presets["None"] = ""
            return presets
        except Exception:
            pass
    return DEFAULT_STYLE_PRESETS.copy()

STYLE_PRESETS = load_style_presets()
state.style_presets = STYLE_PRESETS


def save_image_config(filepath: str, config: dict):
    """Save generation config as JSON sidecar."""
    config_path = Path(filepath).with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_or_create_session_dir(prompt: str = "") -> Path:
    """Get or create session directory for outputs."""
    if state.current_session_dir is None or not state.current_session_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state.session_counter += 1
        safe_name = "".join(c for c in prompt[:25] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        if not safe_name:
            safe_name = "session"
        dir_name = f"session_{timestamp}_{state.session_counter:03d}_{safe_name}"
        state.current_session_dir = OUTPUT_DIR / dir_name
        state.current_session_dir.mkdir(parents=True, exist_ok=True)
    return state.current_session_dir


def generate_image(
    prompt: str,
    negative_prompt: str,
    style: str,
    aspect_ratio: str,
    quality: str,
    seed: int,
    batch_count: int,
    use_ollama: bool,
    ollama_model: str,
    ollama_length: str,
    ollama_complexity: str,
):
    """Generate image(s) with the HunyuanImage model.

    If model is currently loading, waits for it to complete.
    """
    from ui.state import get_state
    app_state = get_state()

    # Wait for model if it's currently loading
    if app_state.model_load_lock.locked():
        yield None, "Model is loading... waiting for it to complete.", "", seed
        # Wait for lock to be released (model finished loading)
        with app_state.model_load_lock:
            pass  # Lock acquired and released = loading finished
        yield None, "Model loaded! Starting generation...", "", seed

    if not is_model_loaded():
        yield None, "Model not loaded. Click 'Load Image Model' first.", "", seed
        return

    model = get_model()
    steps = get_steps_from_quality(quality)
    image_size = get_size_from_aspect(aspect_ratio)

    # Apply style suffix
    style_suffix = STYLE_PRESETS.get(style, "")
    full_prompt = prompt + style_suffix

    # Ollama enhancement
    if use_ollama and state.ollama_available:
        try:
            if state.ollama_enhancer is None:
                state.ollama_enhancer = PromptEnhancer(model=ollama_model)
            enhanced = state.ollama_enhancer.enhance(
                full_prompt, length=ollama_length, complexity=ollama_complexity
            )
            full_prompt = enhanced
            yield None, f"Enhanced prompt with {ollama_model}...", "", seed
        except Exception as e:
            yield None, f"Ollama enhancement failed: {e}", "", seed

    # Resolve wildcards
    if state.wildcard_available and state.wildcard_manager:
        try:
            full_prompt = state.wildcard_manager.process(full_prompt)
        except Exception:
            pass

    # Generate
    session_dir = get_or_create_session_dir(prompt)
    last_image = None
    last_seed = seed

    for i in range(int(batch_count)):
        current_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        if i > 0 and seed < 0:
            current_seed = random.randint(0, 2**31 - 1)

        yield last_image, f"Generating image {i+1}/{int(batch_count)}...", "", current_seed

        try:
            start_time = time.time()

            image = model.generate_image(
                full_prompt,
                current_seed,
                image_size,
                steps,
                stream=True
            )

            gen_time = time.time() - start_time

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
            filename = f"{timestamp}_{current_seed}_{safe_prompt}.png"
            filepath = session_dir / filename
            image.save(str(filepath))

            # Save config
            config = {
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "steps": steps,
                "seed": current_seed,
                "use_ollama": use_ollama,
                "ollama_model": ollama_model if use_ollama else None,
                "generation_time": gen_time,
            }
            save_image_config(str(filepath), config)

            last_image = str(filepath)
            last_seed = current_seed

            info = f"Size: {image_size} | Steps: {steps} | Time: {gen_time:.1f}s"
            yield last_image, f"Generated {i+1}/{int(batch_count)}", info, last_seed

        except Exception as e:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            yield last_image, f"Error on image {i+1}: {e}", "", last_seed

    yield last_image, f"Complete! {int(batch_count)} image(s) saved to {session_dir.name}", f"Size: {image_size} | Steps: {steps}", last_seed


def create_ui():
    """Create the new modular UI."""

    with gr.Blocks(title="HunyuanImage-3.0 v2") as app:

        # Header
        gr.Markdown("""
        # HunyuanImage-3.0 Generator
        **80B MoE Model** | GPU selector at top | Ollama under prompt | Batch-first design
        """)

        # System Bar (GPU + Model controls)
        system = create_system_bar()
        wire_system_bar_events(system)

        # Main two-column layout
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                # Prompt + Ollama controls (grouped!)
                prompt = create_prompt_input(
                    ollama_available=state.ollama_available,
                    wildcard_available=state.wildcard_available
                )
                wire_prompt_input_events(prompt)

                # Generation settings
                settings = create_gen_settings()
                wire_gen_settings_events(settings)

            # Right column - Output
            with gr.Column(scale=1):
                output = create_output_display()
                wire_output_display_events(output)

        # Wire generate button to actual generation
        output.generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt.prompt,
                prompt.negative_prompt,
                prompt.style_dropdown,
                settings.aspect_ratio,
                settings.quality,
                settings.seed,
                settings.batch_count,
                prompt.use_ollama,
                prompt.ollama_model,
                prompt.ollama_length,
                prompt.ollama_complexity,
            ],
            outputs=[
                output.output_image,
                output.status_text,
                output.info_text,
                output.last_seed,
            ]
        )

        # Batch Panel (opens by default)
        batch = create_batch_panel(open_by_default=True)

        # TODO: Wire batch panel events to actual batch generation functions
        # This requires importing the batch generation logic from the original file

        # Recent gallery
        with gr.Accordion("Recent Generations", open=False):
            recent_gallery = gr.Gallery(columns=6, rows=2, height=200)
            refresh_recent_btn = gr.Button("Refresh", size="sm")

        # Footer
        gr.Markdown("""
        ---
        **v2 Layout** | [GitHub](https://github.com/jamesw767/HunyuanImage) |
        Original UI still available as `hunyuan_ui.py`
        """)

    return app


def main():
    print("=" * 60)
    print("HunyuanImage-3.0 Generator - v2 (Modular Layout)")
    print("=" * 60)

    # Initialize GPUs
    init_gpus()

    state_info = get_state()
    print(f"\nGPUs detected: {len(state_info.available_gpus)}")
    for gpu in state_info.available_gpus:
        marker = " <-- selected" if gpu['index'] == state_info.selected_gpu else ""
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.0f} GB){marker}")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Ollama: {'Available' if state_info.ollama_available else 'Not available'}")
    print(f"Wildcards: {'Available' if state_info.wildcard_available else 'Not available'}")

    print("\nStarting web interface on http://localhost:7860")
    print("(Original UI available at hunyuan_ui.py if needed)")

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
