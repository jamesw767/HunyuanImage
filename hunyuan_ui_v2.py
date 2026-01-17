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

# NOTE: We don't set CUDA_VISIBLE_DEVICES here
# Instead we use device_map="cuda:N" to target specific GPUs
# This allows runtime GPU selection without restart

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import random
import json
import gc
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

# Batch manager
try:
    from batch_manager import (
        save_batch_config_named,
        load_batch_config_full,
        get_config_choices,
        refresh_config_dropdown,
        import_batch_from_directory,
    )
    print("[INIT] Batch manager available")
except ImportError as e:
    print(f"[INIT] Batch manager not available: {e}")
    # Provide stubs
    def save_batch_config_named(*args): return "Batch manager not available", gr.update()
    def load_batch_config_full(*args): return [gr.update()] * 16
    def get_config_choices(): return []
    def refresh_config_dropdown(): return gr.update()
    def import_batch_from_directory(*args): return [gr.update()] * 15, "Batch manager not available"

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


def extract_model_name(model_value) -> str:
    """Extract model name from dropdown value (may be dict or string).

    The Ollama model dropdown can return either:
    - A string like "qwen2.5:7b-instruct"
    - A dict like {"name": "qwen2.5:7b-instruct", "size": "4.4GB", ...}
    """
    if not model_value:
        return "qwen2.5:7b-instruct"
    if isinstance(model_value, dict):
        return model_value.get('name', 'qwen2.5:7b-instruct')
    return str(model_value)


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
    import uuid
    gen_id = str(uuid.uuid4())[:8]
    print(f"[GENERATE {gen_id}] Starting generation: prompt='{prompt[:50] if prompt else 'EMPTY'}...'", flush=True)
    print(f"[GENERATE {gen_id}] batch_count={batch_count}, seed={seed}, quality={quality}", flush=True)

    # Check for empty prompt
    if not prompt or not prompt.strip():
        print(f"[GENERATE {gen_id}] Empty prompt - returning early", flush=True)
        yield None, "Please enter a prompt first.", "", seed
        return

    from ui.state import get_state
    app_state = get_state()

    # Wait for model if it's currently loading
    if app_state.model_load_lock.locked():
        print(f"[GENERATE {gen_id}] Model is loading, waiting...", flush=True)
        yield None, "Model is loading... waiting for it to complete.", "", seed
        # Wait for lock to be released (model finished loading)
        with app_state.model_load_lock:
            pass  # Lock acquired and released = loading finished
        yield None, "Model loaded! Starting generation...", "", seed

    if not is_model_loaded():
        print(f"[GENERATE {gen_id}] Model not loaded - returning early", flush=True)
        yield None, "Model not loaded. Click 'Load Image Model' first.", "", seed
        return

    print(f"[GENERATE {gen_id}] Model is loaded, proceeding...", flush=True)

    model = get_model()
    steps = get_steps_from_quality(quality)
    image_size = get_size_from_aspect(aspect_ratio)

    # Apply style suffix
    style_suffix = STYLE_PRESETS.get(style, "")
    full_prompt = prompt + style_suffix

    # Ollama enhancement
    if use_ollama and state.ollama_available:
        try:
            # Extract model name (handles both string and dict from dropdown)
            model_name = extract_model_name(ollama_model)
            length_str = str(ollama_length) if ollama_length else "medium"
            complexity_str = str(ollama_complexity) if ollama_complexity else "detailed"

            if state.ollama_enhancer is None:
                state.ollama_enhancer = PromptEnhancer(model=model_name)
            enhanced = state.ollama_enhancer.enhance(
                full_prompt,
                length=length_str,
                complexity=complexity_str,
                model=model_name  # Pass model to allow runtime switching
            )
            full_prompt = enhanced
            yield None, f"Enhanced prompt with {model_name}...", "", seed
        except ConnectionError as e:
            yield None, f"Ollama server not running. Start with: ollama serve", "", seed
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
    print(f"[GENERATE {gen_id}] Session dir: {session_dir}", flush=True)

    # Write a debug marker file to prove function was called
    debug_marker = session_dir / f"_debug_{gen_id}.txt"
    with open(debug_marker, 'w') as f:
        f.write(f"generate_image called at {datetime.now()}\n")
        f.write(f"prompt: {prompt[:100]}\n")
        f.write(f"batch_count: {batch_count}\n")
        f.write(f"model loaded: {is_model_loaded()}\n")
    print(f"[GENERATE {gen_id}] Debug marker written to {debug_marker}", flush=True)

    last_image = None
    last_seed = seed

    batch_count_int = int(batch_count)
    print(f"[GENERATE {gen_id}] Starting loop for {batch_count_int} images", flush=True)

    for i in range(batch_count_int):
        print(f"[GENERATE {gen_id}] Starting batch iteration {i+1}/{batch_count_int}", flush=True)
        current_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        if i > 0 and seed < 0:
            current_seed = random.randint(0, 2**31 - 1)

        yield last_image, f"Generating image {i+1}/{batch_count_int}...", "", current_seed

        try:
            start_time = time.time()
            print(f"[GENERATE {gen_id}] Calling model.generate_image with size={image_size}, steps={steps}, seed={current_seed}", flush=True)

            image = model.generate_image(
                full_prompt,
                current_seed,
                image_size,
                stream=True,
                diff_infer_steps=steps
            )

            if image is None:
                print(f"[GENERATE {gen_id}] WARNING: model.generate_image returned None!", flush=True)
                yield last_image, f"Error: Model returned no image for iteration {i+1}", "", current_seed
                continue

            gen_time = time.time() - start_time
            print(f"[GENERATE {gen_id}] Image generated in {gen_time:.1f}s, type={type(image)}", flush=True)

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
            filename = f"{timestamp}_{current_seed}_{safe_prompt}.png"
            filepath = session_dir / filename
            print(f"[GENERATE {gen_id}] Saving to {filepath}...", flush=True)
            image.save(str(filepath))
            print(f"[GENERATE {gen_id}] Saved successfully!", flush=True)

            # Save config with all fields needed to recreate
            config = {
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "quality": quality,
                "image_size": image_size,
                "steps": steps,
                "seed": current_seed,
                "batch_count": batch_count,
                "use_ollama": use_ollama,
                "ollama_model": ollama_model if use_ollama else None,
                "ollama_length": ollama_length if use_ollama else None,
                "ollama_complexity": ollama_complexity if use_ollama else None,
                "generation_time": gen_time,
            }
            save_image_config(str(filepath), config)

            last_image = str(filepath)
            last_seed = current_seed

            info = f"Size: {image_size} | Steps: {steps} | Time: {gen_time:.1f}s"
            yield last_image, f"Generated {i+1}/{int(batch_count)}", info, last_seed

        except Exception as e:
            import gc
            import traceback
            error_details = traceback.format_exc()
            print(f"[GENERATE {gen_id} ERROR] {e}", flush=True)
            print(f"[GENERATE {gen_id} TRACEBACK]\n{error_details}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            yield last_image, f"Error on image {i+1}: {e}", "", last_seed

    print(f"[GENERATE {gen_id}] COMPLETE - {batch_count_int} image(s) to {session_dir.name}", flush=True)
    yield last_image, f"Complete! {batch_count_int} image(s) saved to {session_dir.name}", f"Size: {image_size} | Steps: {steps}", last_seed


# ============================================================================
# QUEUE SYSTEM
# ============================================================================

def get_queue_status() -> str:
    """Get current queue status string."""
    app_state = get_state()
    with app_state.queue_lock:
        count = len(app_state.generation_queue)
    if count == 0:
        if app_state.queue_worker_running:
            return "Queue: Processing..."
        return "Queue: 0 pending"
    return f"Queue: {count} pending"


def clear_queue() -> str:
    """Clear all pending items from the queue."""
    app_state = get_state()
    with app_state.queue_lock:
        cleared = len(app_state.generation_queue)
        app_state.generation_queue.clear()
    return f"Queue cleared ({cleared} items removed)"


def add_to_queue(
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
    """Add a generation request to the queue.

    Returns immediately after adding to queue.
    If Ollama is enabled, each queued item will get a fresh enhanced prompt.
    """
    import threading

    if not prompt or not prompt.strip():
        yield None, "Please enter a prompt first.", "", seed, get_queue_status()
        return

    app_state = get_state()

    # Create queue item with all parameters
    queue_item = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'style': style,
        'aspect_ratio': aspect_ratio,
        'quality': quality,
        'seed': seed,
        'batch_count': batch_count,
        'use_ollama': use_ollama,
        'ollama_model': ollama_model,
        'ollama_length': ollama_length,
        'ollama_complexity': ollama_complexity,
    }

    # Add to queue
    with app_state.queue_lock:
        app_state.generation_queue.append(queue_item)
        queue_len = len(app_state.generation_queue)

    print(f"[QUEUE] Added item to queue. Queue length: {queue_len}", flush=True)

    # Start worker if not running
    if not app_state.queue_worker_running:
        print("[QUEUE] Starting queue worker thread...", flush=True)
        worker_thread = threading.Thread(target=queue_worker, daemon=True)
        worker_thread.start()

    yield None, f"Added to queue (position {queue_len})", "", seed, get_queue_status()


def queue_worker():
    """Background worker that processes the generation queue."""
    import uuid

    app_state = get_state()
    app_state.queue_worker_running = True
    app_state.queue_stop_requested = False

    print("[QUEUE WORKER] Started", flush=True)

    try:
        while True:
            # Check for stop request
            if app_state.queue_stop_requested:
                print("[QUEUE WORKER] Stop requested, exiting", flush=True)
                break

            # Get next item from queue
            with app_state.queue_lock:
                if not app_state.generation_queue:
                    print("[QUEUE WORKER] Queue empty, exiting", flush=True)
                    break
                item = app_state.generation_queue.pop(0)
                remaining = len(app_state.generation_queue)

            print(f"[QUEUE WORKER] Processing item. {remaining} remaining in queue", flush=True)

            # Process this item
            try:
                process_queue_item(item)
            except Exception as e:
                print(f"[QUEUE WORKER] Error processing item: {e}", flush=True)
                import traceback
                traceback.print_exc()

    finally:
        app_state.queue_worker_running = False
        print("[QUEUE WORKER] Stopped", flush=True)


def process_queue_item(item: dict):
    """Process a single queue item - generates image with fresh Ollama enhancement."""
    import uuid
    gen_id = str(uuid.uuid4())[:8]

    prompt = item['prompt']
    style = item['style']
    use_ollama = item['use_ollama']
    ollama_model = item['ollama_model']
    ollama_length = item['ollama_length']
    ollama_complexity = item['ollama_complexity']
    quality = item['quality']
    aspect_ratio = item['aspect_ratio']
    seed = item['seed']
    batch_count = item['batch_count']

    print(f"[QUEUE ITEM {gen_id}] Starting: '{prompt[:50]}...'", flush=True)

    app_state = get_state()

    # Check model loaded
    if not is_model_loaded():
        print(f"[QUEUE ITEM {gen_id}] Model not loaded, skipping", flush=True)
        return

    model = get_model()
    steps = get_steps_from_quality(quality)
    image_size = get_size_from_aspect(aspect_ratio)

    # Apply style suffix
    style_suffix = STYLE_PRESETS.get(style, "")
    full_prompt = prompt + style_suffix

    # Ollama enhancement - FRESH enhancement for each queue item!
    if use_ollama and state.ollama_available:
        try:
            # Extract model name (handles both string and dict from dropdown)
            model_name = extract_model_name(ollama_model)
            length_str = str(ollama_length) if ollama_length else "medium"
            complexity_str = str(ollama_complexity) if ollama_complexity else "detailed"

            if state.ollama_enhancer is None:
                from ollama_prompts import PromptEnhancer
                state.ollama_enhancer = PromptEnhancer(model=model_name)

            print(f"[QUEUE ITEM {gen_id}] Enhancing prompt with {model_name}...", flush=True)
            enhanced = state.ollama_enhancer.enhance(
                full_prompt,
                length=length_str,
                complexity=complexity_str,
                model=model_name
            )
            full_prompt = enhanced
            print(f"[QUEUE ITEM {gen_id}] Enhanced prompt: '{full_prompt[:100]}...'", flush=True)
        except Exception as e:
            print(f"[QUEUE ITEM {gen_id}] Ollama enhancement failed: {e}", flush=True)

    # Resolve wildcards
    if state.wildcard_available and state.wildcard_manager:
        try:
            full_prompt = state.wildcard_manager.process(full_prompt)
        except Exception:
            pass

    # Generate
    session_dir = get_or_create_session_dir(prompt)

    batch_count_int = int(batch_count)
    for i in range(batch_count_int):
        if app_state.queue_stop_requested:
            print(f"[QUEUE ITEM {gen_id}] Stop requested during batch", flush=True)
            break

        current_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
        if i > 0 and seed < 0:
            current_seed = random.randint(0, 2**31 - 1)

        try:
            print(f"[QUEUE ITEM {gen_id}] Generating image {i+1}/{batch_count_int}...", flush=True)

            image = model.generate_image(
                full_prompt,
                current_seed,
                image_size,
                stream=True,
                diff_infer_steps=steps
            )

            if image is None:
                print(f"[QUEUE ITEM {gen_id}] Model returned None", flush=True)
                continue

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
            filename = f"{timestamp}_{current_seed}_{safe_prompt}.png"
            filepath = session_dir / filename
            image.save(str(filepath))
            print(f"[QUEUE ITEM {gen_id}] Saved: {filename}", flush=True)

            # Save config with all fields needed to recreate
            config = {
                "prompt": prompt,
                "full_prompt": full_prompt,
                "negative_prompt": item.get('negative_prompt', ''),
                "style": style,
                "aspect_ratio": aspect_ratio,
                "quality": quality,
                "image_size": image_size,
                "steps": steps,
                "seed": current_seed,
                "batch_count": batch_count,
                "use_ollama": use_ollama,
                "ollama_model": ollama_model if use_ollama else None,
                "ollama_length": ollama_length if use_ollama else None,
                "ollama_complexity": ollama_complexity if use_ollama else None,
                "queued": True,
            }
            save_image_config(str(filepath), config)

            # Clean up
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[QUEUE ITEM {gen_id}] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"[QUEUE ITEM {gen_id}] Complete", flush=True)


def stop_queue() -> str:
    """Stop queue processing after current item."""
    app_state = get_state()
    app_state.queue_stop_requested = True
    return "Stopping queue after current item..."


def load_settings_from_json(file_obj):
    """Load generation settings from a JSON config file.

    Returns tuple of all settings to populate the UI fields.
    """
    if file_obj is None:
        return (
            gr.update(),  # prompt
            gr.update(),  # negative_prompt
            gr.update(),  # style
            gr.update(),  # aspect_ratio
            gr.update(),  # quality
            gr.update(),  # seed
            gr.update(),  # batch_count
            gr.update(),  # use_ollama
            gr.update(),  # ollama_model
            gr.update(),  # ollama_length
            gr.update(),  # ollama_complexity
            "No file selected",
        )

    try:
        # Read the JSON file
        if hasattr(file_obj, 'name'):
            file_path = file_obj.name
        else:
            file_path = str(file_obj)

        with open(file_path, 'r') as f:
            config = json.load(f)

        # Extract settings with defaults (handle both single image and batch field names)
        prompt = config.get('prompt', '')
        negative_prompt = config.get('negative_prompt', '')
        style = config.get('style', 'None')
        aspect_ratio = config.get('aspect_ratio', '1:1 (Square)')
        # Handle both 'quality' and 'quality_preset' field names
        quality = config.get('quality') or config.get('quality_preset', 'Standard (20 steps)')
        seed = config.get('seed', -1)
        batch_count = config.get('batch_count', 1)
        # Handle both 'use_ollama' and 'enhance_prompts' field names
        use_ollama = config.get('use_ollama') if 'use_ollama' in config else config.get('enhance_prompts', False)
        ollama_model = config.get('ollama_model', 'qwen2.5:7b-instruct')
        ollama_length = config.get('ollama_length', 'medium')
        ollama_complexity = config.get('ollama_complexity', 'detailed')

        status = f"Loaded settings (seed: {seed})"

        return (
            prompt,
            negative_prompt,
            style,
            aspect_ratio,
            quality,
            seed,
            batch_count,
            use_ollama,
            ollama_model if ollama_model else 'qwen2.5:7b-instruct',
            ollama_length if ollama_length else 'medium',
            ollama_complexity if ollama_complexity else 'detailed',
            status,
        )

    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            f"Error loading settings: {e}",
        )


# ============================================================================
# BATCH PROCESSING FUNCTIONS
# ============================================================================

def calculate_batch_preview(
    themes_text: str,
    variations: int,
    images_per: int,
    styles: list,
) -> str:
    """Calculate and preview batch generation count."""
    if not themes_text.strip():
        return "Enter themes to see preview..."

    themes = [t.strip() for t in themes_text.strip().split('\n') if t.strip()]
    num_themes = len(themes)
    num_styles = len(styles) if styles else 1

    total_variations = num_themes * variations
    total_images = total_variations * num_styles * images_per

    preview = f"""**Batch Preview:**
- **Themes:** {num_themes}
- **Variations per theme:** {variations}
- **Styles:** {num_styles} ({', '.join(styles[:3])}{'...' if len(styles) > 3 else ''})
- **Images per combo:** {images_per}

**Total: {total_images} images**
"""
    return preview


def start_batch_generation(
    batch_name: str,
    themes_text: str,
    negative_prompt: str,
    variations: int,
    images_per: int,
    styles: list,
    aspect_ratio: str,
    quality: str,
    guidance_scale: float,
    llm_backend: str,
    enhance: bool,
    ollama_model: str,
    ollama_length: str,
    ollama_complexity: str,
    random_seeds: bool,
):
    """Start batch generation process."""
    print(f"[BATCH] ========== START BATCH CALLED ==========", flush=True)

    # IMMEDIATELY reset state flags before anything else
    app_state = get_state()
    app_state.batch_stop_requested = False
    app_state.batch_running = False  # Will set to True after validation
    print(f"[BATCH] State reset - stop_requested: False", flush=True)

    if not themes_text.strip():
        print("[BATCH] No themes provided", flush=True)
        yield "No themes provided", "", []
        return

    themes = [t.strip() for t in themes_text.strip().split('\n') if t.strip()]
    print(f"[BATCH] Themes parsed: {len(themes)}", flush=True)

    if not styles:
        styles = ["None"]
    print(f"[BATCH] Styles: {styles}", flush=True)

    if not is_model_loaded():
        print("[BATCH] Model not loaded", flush=True)
        yield "Model not loaded. Load the model first.", "", []
        return
    print(f"[BATCH] Model is loaded", flush=True)

    # Create batch output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in batch_name[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    batch_dir = OUTPUT_DIR / "batches" / f"{safe_name}_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    app_state.batch_running = True
    app_state.batch_stop_requested = False
    app_state.batch_results = []

    print(f"[BATCH] Flags set - running: True, stop_requested: False", flush=True)
    print(f"[BATCH] Starting: {len(themes)} themes x {len(styles)} styles x {variations} vars x {images_per} imgs", flush=True)

    yield f"Starting batch: {len(themes)} themes x {len(styles)} styles", str(batch_dir), []

    model = get_model()
    if model is None:
        print("[BATCH] ERROR: get_model() returned None!", flush=True)
        yield "Error: Model not available", str(batch_dir), []
        return

    steps = get_steps_from_quality(quality)
    image_size = get_size_from_aspect(aspect_ratio)
    print(f"[BATCH] Model ready, steps={steps}, size={image_size}", flush=True)

    total_count = 0
    generated_images = []

    # Round-robin iteration: cycle through all themes/styles before next iteration
    # This gives variety instead of doing all variations of one theme first
    total_iterations = int(variations)
    total_images_per = int(images_per)

    for iteration_idx in range(total_iterations):
        print(f"[BATCH] === Iteration {iteration_idx+1}/{total_iterations} ===", flush=True)
        if app_state.batch_stop_requested:
            break

        for theme_idx, theme in enumerate(themes):
            if app_state.batch_stop_requested:
                break

            for style in styles:
                if app_state.batch_stop_requested:
                    break

                for img_idx in range(total_images_per):
                    if app_state.batch_stop_requested:
                        print(f"[BATCH] Stop requested, breaking", flush=True)
                        break

                    print(f"[BATCH] -- Iteration {iteration_idx+1}, Theme {theme_idx+1}/{len(themes)}, Style: {style}, Image {img_idx+1}/{total_images_per}", flush=True)

                    # Build prompt
                    full_prompt = theme
                    style_suffix = STYLE_PRESETS.get(style, "")
                    if style_suffix:
                        full_prompt += style_suffix

                    # Enhance with Ollama if enabled
                    if enhance and state.ollama_available:
                        print(f"[BATCH] Enhancing with Ollama...", flush=True)
                        try:
                            # Extract model name, but ALWAYS prefer small model for batch
                            model_name = extract_model_name(ollama_model)

                            # Force small model for batch to avoid loading huge models
                            if "30b" in model_name.lower() or "24b" in model_name.lower() or "20b" in model_name.lower():
                                model_name = "qwen2.5:7b-instruct"
                                print(f"[BATCH] Forcing small model: {model_name}", flush=True)

                            # Check VRAM before loading (use nvidia-smi index for Ollama GPU)
                            try:
                                from ollama_prompts import check_model_fits_gpu
                                # Find nvidia-smi index for selected Ollama GPU
                                ollama_pytorch_gpu = app_state.selected_ollama_gpu
                                ollama_nvidia_gpu = ollama_pytorch_gpu  # default
                                for gpu in app_state.available_gpus:
                                    if gpu['index'] == ollama_pytorch_gpu:
                                        ollama_nvidia_gpu = gpu.get('nvidia_index', ollama_pytorch_gpu)
                                        break
                                fits, vram_msg = check_model_fits_gpu(model_name, gpu_index=ollama_nvidia_gpu)
                                print(f"[BATCH] VRAM check (nvidia GPU {ollama_nvidia_gpu}): {vram_msg}", flush=True)
                                if not fits:
                                    yield f"⚠️ {vram_msg}", str(batch_dir), generated_images
                            except Exception as vram_err:
                                print(f"[BATCH] VRAM check error: {vram_err}", flush=True)

                            if state.ollama_enhancer is None:
                                state.ollama_enhancer = PromptEnhancer(model=model_name)

                            # Handle random length/complexity selection
                            actual_length = ollama_length
                            actual_complexity = ollama_complexity
                            if str(ollama_length).lower() == "random":
                                length_choices = ["minimal", "short", "medium", "long", "detailed", "cinematic", "experimental"]
                                actual_length = random.choice(length_choices)
                                print(f"[BATCH] Random length selected: {actual_length}", flush=True)
                            if str(ollama_complexity).lower() == "random":
                                complexity_choices = ["simple", "moderate", "detailed", "complex", "cinematic", "experimental"]
                                actual_complexity = random.choice(complexity_choices)
                                print(f"[BATCH] Random complexity selected: {actual_complexity}", flush=True)

                            # Show loading status
                            yield f"Loading Ollama model {model_name}...", str(batch_dir), generated_images

                            full_prompt = state.ollama_enhancer.enhance(
                                full_prompt,
                                length=str(actual_length) if actual_length else "medium",
                                complexity=str(actual_complexity) if actual_complexity else "detailed",
                                model=model_name
                            )
                            print(f"[BATCH] Ollama done: {full_prompt[:80]}...", flush=True)
                        except Exception as e:
                            print(f"[BATCH] Ollama enhancement failed: {e}", flush=True)
                            yield f"Ollama error: {str(e)[:100]}", str(batch_dir), generated_images

                    # Resolve wildcards
                    if state.wildcard_available and state.wildcard_manager:
                        try:
                            full_prompt = state.wildcard_manager.process(full_prompt)
                        except Exception:
                            pass

                    # Generate seed
                    current_seed = random.randint(0, 2**31 - 1) if random_seeds else (iteration_idx * 10000 + theme_idx * 1000 + img_idx)
                    print(f"[BATCH] Seed: {current_seed}, prompt length: {len(full_prompt)}", flush=True)

                    try:
                        print(f"[BATCH] Yielding status update...", flush=True)
                        yield f"Iter {iteration_idx+1}/{total_iterations} | #{total_count + 1}: {theme[:25]}... ({style})", str(batch_dir), generated_images

                        print(f"[BATCH] Calling model.generate_image()...", flush=True)
                        start_time = time.time()
                        image = model.generate_image(
                            full_prompt,
                            current_seed,
                            image_size,
                            stream=True,
                            diff_infer_steps=steps
                        )
                        gen_time = time.time() - start_time
                        print(f"[BATCH] generate_image returned: {type(image)} in {gen_time:.1f}s", flush=True)

                        if image:
                            filename = f"{total_count:04d}_{current_seed}_{style.replace(' ', '_')[:20]}.png"
                            filepath = batch_dir / filename
                            image.save(str(filepath))

                            # Save config (matching single image format)
                            config = {
                                "prompt": theme,
                                "full_prompt": full_prompt,
                                "negative_prompt": negative_prompt,
                                "style": style,
                                "aspect_ratio": aspect_ratio,
                                "quality": quality,
                                "image_size": image_size,
                                "steps": steps,
                                "seed": current_seed,
                                "use_ollama": enhance,
                                "ollama_model": ollama_model if enhance else None,
                                "ollama_length": ollama_length if enhance else None,
                                "ollama_complexity": ollama_complexity if enhance else None,
                                "generation_time": gen_time,
                                "batch_name": batch_name,
                                "batch_iteration": iteration_idx + 1,
                                "batch_total_iterations": total_iterations,
                            }
                            save_image_config(str(filepath), config)

                            generated_images.append(str(filepath))
                            total_count += 1

                            # Clean up
                            gc.collect()
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"[BATCH] Error: {e}")

    app_state.batch_running = False
    yield f"Batch complete! Generated {total_count} images", str(batch_dir), generated_images


def stop_batch():
    """Stop current batch generation."""
    app_state = get_state()
    app_state.batch_stop_requested = True
    app_state.batch_running = False  # Also reset running flag
    print("[BATCH] Stop requested - flags reset", flush=True)
    return "Batch stopped."


# ============================================================================
# WILDCARD EDITOR FUNCTIONS
# ============================================================================

def get_wildcard_categories():
    """Get list of wildcard categories."""
    if state.wildcard_manager:
        return state.wildcard_manager.get_available_wildcards()
    return []


def load_wildcard_category(category: str):
    """Load items from a wildcard category."""
    if not category or not state.wildcard_manager:
        return "", f"No category selected"

    items = state.wildcard_manager.data.get(category, [])
    items_text = "\n".join(items)
    info = f"{len(items)} items in [{category}]"
    return items_text, info


def save_wildcard_category(category: str, items_text: str):
    """Save items to a wildcard category."""
    if not category or not state.wildcard_manager:
        return "No category selected"

    items = [i.strip() for i in items_text.strip().split('\n') if i.strip()]
    state.wildcard_manager.data[category] = items
    return f"Updated [{category}] with {len(items)} items (not saved to file yet)"


def add_wildcard_item(category: str, items_text: str, new_item: str):
    """Add an item to a wildcard category."""
    if not new_item.strip():
        return items_text, "Enter an item to add"

    items = items_text.strip().split('\n') if items_text.strip() else []
    items.append(new_item.strip())
    return "\n".join(items), f"Added: {new_item.strip()}"


def create_wildcard_category(category_name: str):
    """Create a new wildcard category."""
    if not category_name.strip():
        return gr.update(), "Enter a category name"

    if not state.wildcard_manager:
        return gr.update(), "Wildcard manager not available"

    safe_name = category_name.strip().lower().replace(' ', '-')
    if safe_name in state.wildcard_manager.data:
        return gr.update(), f"Category [{safe_name}] already exists"

    state.wildcard_manager.data[safe_name] = []
    categories = get_wildcard_categories()
    return gr.update(choices=categories, value=safe_name), f"Created category [{safe_name}]"


def delete_wildcard_category(category: str):
    """Delete a wildcard category."""
    if not category or not state.wildcard_manager:
        return gr.update(), "", "No category selected"

    if category in state.wildcard_manager.data:
        del state.wildcard_manager.data[category]
        categories = get_wildcard_categories()
        return gr.update(choices=categories, value=categories[0] if categories else None), "", f"Deleted [{category}]"

    return gr.update(), "", f"Category [{category}] not found"


def save_wildcards_to_file():
    """Save all wildcard changes to file."""
    if not state.wildcard_manager:
        return "Wildcard manager not available"

    try:
        with open(state.wildcard_manager.json_path, 'w', encoding='utf-8') as f:
            json.dump(state.wildcard_manager.data, f, indent=2, ensure_ascii=False)
        return f"Saved {len(state.wildcard_manager.data)} categories to {state.wildcard_manager.json_path}"
    except Exception as e:
        return f"Error saving: {e}"


def reload_wildcards():
    """Reload wildcards from file."""
    if state.wildcard_manager:
        state.wildcard_manager.reload()
        categories = get_wildcard_categories()
        return gr.update(choices=categories), f"Reloaded {len(categories)} categories"
    return gr.update(), "Wildcard manager not available"


def import_wildcards_from_source():
    """Import wildcards from sd-wildcards/wildcards/*.txt files."""
    source_dir = Path(__file__).parent / "sd-wildcards" / "wildcards"

    if not source_dir.exists():
        return gr.update(), f"Source directory not found: {source_dir}"

    if not state.wildcard_manager:
        return gr.update(), "Wildcard manager not available"

    imported_count = 0
    total_items = 0

    # Clear existing data and rebuild from source files
    state.wildcard_manager.data = {}

    for txt_file in sorted(source_dir.glob("*.txt")):
        category_name = txt_file.stem  # filename without .txt

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                items = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            if items:
                state.wildcard_manager.data[category_name] = items
                imported_count += 1
                total_items += len(items)
        except Exception as e:
            print(f"[WILDCARD IMPORT] Error reading {txt_file}: {e}")

    # Save to JSON
    try:
        with open(state.wildcard_manager.json_path, 'w', encoding='utf-8') as f:
            json.dump(state.wildcard_manager.data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return gr.update(), f"Imported but failed to save: {e}"

    categories = get_wildcard_categories()
    return gr.update(choices=categories), f"Imported {imported_count} categories with {total_items} total items"


def search_wildcard_categories(query: str):
    """Filter wildcard categories by search query."""
    if not state.wildcard_manager:
        return gr.update()

    all_categories = get_wildcard_categories()
    if not query.strip():
        return gr.update(choices=all_categories)

    filtered = [c for c in all_categories if query.lower() in c.lower()]
    return gr.update(choices=filtered)


def generate_wildcards_from_ollama(prompt: str, count: int, complexity: str = "medium") -> Tuple[str, str]:
    """Use Ollama LLM to generate a list of items for a wildcard category.

    Args:
        prompt: What to generate (e.g., "metals", "D&D characters", "dungeon environments")
        count: Number of items to generate
        complexity: Detail level - "keywords", "short", "medium", "long", "cinematic"

    Returns:
        Tuple of (items_text, status_message)
    """
    if not prompt or not prompt.strip():
        return "", "Enter what kind of items to generate"

    if not state.ollama_available or not state.ollama_manager:
        return "", "Ollama is not available"

    if not state.ollama_manager.is_running():
        return "", "Ollama server is not running. Start it first."

    try:
        # Get a model to use
        models = state.ollama_manager.list_models()
        if not models:
            return "", "No Ollama models available. Pull a model first."

        # Extract model names from dicts (ollama_manager returns dicts with 'name' key)
        model_names = []
        for m in models:
            if isinstance(m, dict):
                model_names.append(m.get('name', ''))
            else:
                model_names.append(str(m))

        # Prefer small models for quick wildcard generation
        # Priority: qwen2.5:7b-instruct > any 7b/8b model > smallest available
        model = None

        # First: exact match for preferred small model
        if "qwen2.5:7b-instruct" in model_names:
            model = "qwen2.5:7b-instruct"

        # Second: look for any small model (7b or 8b)
        if not model:
            for name in model_names:
                name_lower = name.lower()
                if "7b" in name_lower or "8b" in name_lower:
                    model = name
                    break

        # Fallback: first available model
        if not model and model_names:
            model = model_names[0]

        if not model:
            return "", "No suitable model found"

        # Complexity-based prompts for different detail levels
        complexity_configs = {
            "keywords": {
                "system": """You generate simple keyword lists for image prompts.
Return ONLY single words or 2-word phrases, one per line.
No numbering, no explanations, no formatting.
Be diverse and creative.""",
                "user": f"""Generate exactly {int(count)} unique keywords for: "{prompt.strip()}"
Single words or short 2-word phrases only. One per line.""",
                "max_tokens": 1000
            },
            "short": {
                "system": """You generate short descriptive phrases for image prompts.
Return phrases of 3-6 words each, one per line.
No numbering, no explanations.
Be vivid and specific.""",
                "user": f"""Generate exactly {int(count)} unique short descriptions for: "{prompt.strip()}"
Each should be 3-6 words. One per line, no numbers.""",
                "max_tokens": 1500
            },
            "medium": {
                "system": """You generate descriptive scene fragments for image prompts.
Return phrases of 8-15 words each, one per line.
Include visual details like lighting, atmosphere, materials.
No numbering, no explanations.""",
                "user": f"""Generate exactly {int(count)} unique scene descriptions for: "{prompt.strip()}"
Each should be 8-15 words with visual details. One per line, no numbers.""",
                "max_tokens": 3000
            },
            "long": {
                "system": """You generate detailed scene descriptions for AI image generation.
Return rich, evocative descriptions of 20-40 words each, one per line.
Include: subject, environment, lighting, mood, colors, textures, atmosphere.
Write in descriptive prose style. No numbering, no explanations.""",
                "user": f"""Generate exactly {int(count)} unique detailed scene descriptions for: "{prompt.strip()}"

Each description should be 20-40 words and include:
- Main subject or focal point
- Environmental details
- Lighting and atmosphere
- Colors and textures
- Mood or emotional tone

One complete description per line, no numbers or bullets.""",
                "max_tokens": 5000
            },
            "cinematic": {
                "system": """You are a cinematic prompt writer for AI image generation.
Create vivid, film-quality scene descriptions like a cinematographer would describe a shot.
Each description should be 40-80 words, rich with visual storytelling.
Include: camera angle, lighting style, color palette, atmosphere, focal elements, mood.
Write in evocative prose. No numbering, no explanations, just the descriptions.""",
                "user": f"""Generate exactly {int(count)} unique cinematic scene descriptions for: "{prompt.strip()}"

Each should be a complete visual scene (40-80 words) including:
- Camera perspective and framing
- Dramatic lighting (golden hour, neon glow, volumetric rays, etc.)
- Color palette and visual mood
- Environmental atmosphere and weather
- Key visual elements and their arrangement
- Emotional tone and narrative suggestion

Write like describing shots for a film. One complete scene per line.""",
                "max_tokens": 8000
            }
        }

        # Get config for selected complexity (default to medium)
        config = complexity_configs.get(complexity, complexity_configs["medium"])

        # Call Ollama API directly
        from ollama_prompts import OllamaClient
        client = OllamaClient()

        response = client.generate(
            prompt=config["user"],
            model=model,
            system=config["system"],
            temperature=0.85,
            max_tokens=config["max_tokens"]
        )

        if response and response.text:
            raw_text = response.text
            # Clean up the response - filter out empty lines and numbered items
            lines = []
            for line in raw_text.strip().split('\n'):
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Remove leading numbers like "1." or "1)" or "- "
                import re
                cleaned = re.sub(r'^[\d]+[\.\)\-\s]+', '', line).strip()
                cleaned = re.sub(r'^[\-\*\•]\s*', '', cleaned).strip()
                if cleaned:
                    lines.append(cleaned)

            items_text = '\n'.join(lines)
            actual_count = len(lines)
            return items_text, f"Generated {actual_count} items for '{prompt}' using {model}"
        else:
            return "", f"No response from Ollama"

    except Exception as e:
        return "", f"Error generating from Ollama: {str(e)}"


def generate_and_save_wildcard_category(prompt: str, count: int, complexity: str = "medium"):
    """Generate wildcards from Ollama and save as a new category.

    Args:
        prompt: Category name / what to generate (e.g., "dungeon environments")
        count: Number of items to generate
        complexity: Detail level - "keywords", "short", "medium", "long", "cinematic"

    Returns:
        Tuple of (dropdown_update, items_text, status_message)
    """
    if not prompt or not prompt.strip():
        return gr.update(), "", "Enter a category name / what to generate"

    # Generate the items with specified complexity
    items_text, gen_status = generate_wildcards_from_ollama(prompt, count, complexity)

    if not items_text:
        return gr.update(), "", gen_status

    # Create category name from prompt (sanitize)
    category_name = prompt.strip().lower().replace(" ", "-")
    category_name = "".join(c for c in category_name if c.isalnum() or c == "-")

    if not category_name:
        return gr.update(), items_text, f"Generated items but invalid category name"

    # Check if category exists
    if state.wildcard_manager:
        existing = state.wildcard_manager.get_available_wildcards()
        if category_name in existing:
            # Add to existing category
            existing_items = state.wildcard_manager.data.get(category_name, [])
            new_items = [line.strip() for line in items_text.split('\n') if line.strip()]
            combined = list(set(existing_items + new_items))  # Dedupe
            state.wildcard_manager.data[category_name] = combined
            items_text = '\n'.join(combined)
            status = f"Added {len(new_items)} items to existing '{category_name}' (total: {len(combined)})"
        else:
            # Create new category
            new_items = [line.strip() for line in items_text.split('\n') if line.strip()]
            state.wildcard_manager.data[category_name] = new_items
            status = f"Created category '{category_name}' with {len(new_items)} items"

        # Save to file
        try:
            import json
            with open(state.wildcard_manager.json_path, 'w', encoding='utf-8') as f:
                json.dump(state.wildcard_manager.data, f, indent=2, ensure_ascii=False)
            status += " (saved)"
        except Exception as e:
            status += f" (save failed: {e})"

        # Update dropdown
        categories = get_wildcard_categories()
        return gr.update(choices=categories, value=category_name), items_text, status

    return gr.update(), items_text, "Wildcard manager not available"


# ============================================================================
# STYLE EDITOR FUNCTIONS
# ============================================================================

def get_style_names():
    """Get list of style preset names."""
    return list(STYLE_PRESETS.keys())


def load_style(style_name: str):
    """Load a style preset for editing."""
    if not style_name:
        return "", "", "Select a style"

    suffix = STYLE_PRESETS.get(style_name, "")
    preview = f"**{style_name}**\n\nSuffix: `{suffix[:100]}{'...' if len(suffix) > 100 else ''}`"
    return style_name, suffix, preview


def save_style(style_name: str, suffix: str):
    """Save changes to a style preset."""
    if not style_name:
        return "No style selected"

    STYLE_PRESETS[style_name] = suffix
    state.style_presets[style_name] = suffix
    return f"Updated style: {style_name}"


def delete_style(style_name: str):
    """Delete a style preset."""
    if not style_name or style_name == "None":
        return gr.update(), "Cannot delete 'None' style"

    if style_name in STYLE_PRESETS:
        del STYLE_PRESETS[style_name]
    if style_name in state.style_presets:
        del state.style_presets[style_name]

    styles = get_style_names()
    return gr.update(choices=styles, value=styles[0] if styles else None), f"Deleted style: {style_name}"


def create_style(name: str, suffix: str):
    """Create a new style preset."""
    if not name.strip():
        return gr.update(), "Enter a style name"

    if name in STYLE_PRESETS:
        return gr.update(), f"Style '{name}' already exists"

    STYLE_PRESETS[name] = suffix
    state.style_presets[name] = suffix
    styles = get_style_names()
    return gr.update(choices=styles, value=name), f"Created style: {name}"


def save_styles_to_file():
    """Save style presets to a JSON file."""
    styles_path = OUTPUT_DIR / "style_presets.json"
    try:
        with open(styles_path, 'w') as f:
            json.dump(STYLE_PRESETS, f, indent=2)
        return f"Saved {len(STYLE_PRESETS)} styles to {styles_path}"
    except Exception as e:
        return f"Error saving: {e}"


def reload_styles():
    """Reload style presets."""
    styles = get_style_names()
    return gr.update(choices=styles), f"Loaded {len(styles)} styles"


def refresh_latest_image():
    """Get the most recently generated image and queue status."""
    app_state = get_state()

    # Find most recent image in current session or output dir
    latest_image = None
    latest_time = 0

    search_dir = app_state.current_session_dir if app_state.current_session_dir else OUTPUT_DIR
    if search_dir and search_dir.exists():
        for img_path in search_dir.glob("**/*.png"):
            try:
                mtime = img_path.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_image = str(img_path)
            except Exception:
                pass

    queue_status = get_queue_status()
    status_msg = "Refreshed" if latest_image else "No images found"

    return latest_image, status_msg, queue_status


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

        # Wire generate button to queue system
        output.generate_btn.click(
            fn=add_to_queue,
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
                output.queue_status,
            ]
        )

        # Wire stop button to stop queue
        output.stop_btn.click(
            fn=stop_queue,
            inputs=[],
            outputs=[output.status_text]
        )

        # Wire clear queue button
        output.clear_queue_btn.click(
            fn=clear_queue,
            inputs=[],
            outputs=[output.queue_status]
        )

        # Wire refresh button to show latest image and queue status
        output.refresh_btn.click(
            fn=refresh_latest_image,
            inputs=[],
            outputs=[output.output_image, output.status_text, output.queue_status]
        )

        # Wire load settings button to populate all fields from JSON
        prompt.load_settings_btn.click(
            fn=load_settings_from_json,
            inputs=[prompt.settings_file],
            outputs=[
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
                output.status_text,
            ]
        )

        # Also auto-load when file is dropped
        prompt.settings_file.change(
            fn=load_settings_from_json,
            inputs=[prompt.settings_file],
            outputs=[
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
                output.status_text,
            ]
        )

        # Batch Panel (opens by default)
        batch = create_batch_panel(open_by_default=True)

        # Wire batch panel events
        # Calculate preview
        batch.config.batch_calculate_btn.click(
            fn=calculate_batch_preview,
            inputs=[
                batch.config.batch_themes,
                batch.config.batch_variations,
                batch.config.batch_images_per,
                batch.config.batch_styles,
            ],
            outputs=[batch.config.batch_preview]
        )

        # Start batch
        batch.config.batch_start_btn.click(
            fn=start_batch_generation,
            inputs=[
                batch.config.batch_name,
                batch.config.batch_themes,
                batch.config.batch_negative_prompt,
                batch.config.batch_variations,
                batch.config.batch_images_per,
                batch.config.batch_styles,
                batch.config.batch_aspect,
                batch.config.batch_quality,
                batch.config.batch_guidance_scale,
                batch.config.batch_llm_backend,
                batch.config.batch_enhance,
                batch.config.batch_ollama_model,
                batch.config.batch_ollama_length,
                batch.config.batch_ollama_complexity,
                batch.config.batch_random_seeds,
            ],
            outputs=[
                batch.config.batch_status,
                batch.config.batch_output_dir,
                batch.config.batch_gallery,
            ]
        )

        # Stop batch
        batch.config.batch_stop_btn.click(
            fn=stop_batch,
            inputs=[],
            outputs=[batch.config.batch_status]
        )

        # Save batch config
        batch.config.batch_save_btn.click(
            fn=lambda *args: save_batch_config_named(*args),
            inputs=[
                batch.config.config_name_input,
                batch.config.batch_name,
                batch.config.batch_themes,
                batch.config.batch_variations,
                batch.config.batch_styles,
                batch.config.batch_images_per,
                batch.config.batch_llm_backend,
                batch.config.batch_ollama_model,
                batch.config.batch_enhance,
                batch.config.batch_aspect,
                batch.config.batch_quality,
                batch.config.batch_random_seeds,
                batch.config.batch_negative_prompt,
                batch.config.batch_ollama_length,
                batch.config.batch_ollama_complexity,
            ],
            outputs=[batch.config.batch_config_status, batch.config.batch_config_dropdown]
        )

        # Load batch config
        batch.config.batch_load_btn.click(
            fn=load_batch_config_full,
            inputs=[batch.config.batch_config_dropdown],
            outputs=[
                batch.config.batch_name,
                batch.config.batch_themes,
                batch.config.batch_variations,
                batch.config.batch_styles,
                batch.config.batch_images_per,
                batch.config.batch_llm_backend,
                batch.config.batch_ollama_model,
                batch.config.batch_enhance,
                batch.config.batch_aspect,
                batch.config.batch_quality,
                batch.config.batch_random_seeds,
                batch.config.batch_negative_prompt,
                batch.config.batch_ollama_length,
                batch.config.batch_ollama_complexity,
                batch.config.config_name_input,
                batch.config.batch_config_status,
            ]
        )

        # Refresh batch configs dropdown
        batch.config.refresh_configs_btn.click(
            fn=refresh_config_dropdown,
            outputs=[batch.config.batch_config_dropdown]
        )

        # Import batch from directory
        def import_from_dir_wrapper(dir_path):
            """Wrapper to unpack the tuple from import_batch_from_directory"""
            updates, status = import_batch_from_directory(dir_path)
            return updates + [status]

        batch.config.import_dir_btn.click(
            fn=import_from_dir_wrapper,
            inputs=[batch.config.import_dir_path],
            outputs=[
                batch.config.batch_name,
                batch.config.batch_themes,
                batch.config.batch_variations,
                batch.config.batch_styles,
                batch.config.batch_images_per,
                batch.config.batch_llm_backend,
                batch.config.batch_ollama_model,
                batch.config.batch_enhance,
                batch.config.batch_aspect,
                batch.config.batch_quality,
                batch.config.batch_random_seeds,
                batch.config.batch_negative_prompt,
                batch.config.batch_ollama_length,
                batch.config.batch_ollama_complexity,
                batch.config.config_name_input,
                batch.config.batch_config_status,
            ]
        )

        # Wire Wildcard Editor
        # Search wildcards
        batch.wildcards.wildcard_search.change(
            fn=search_wildcard_categories,
            inputs=[batch.wildcards.wildcard_search],
            outputs=[batch.wildcards.wildcard_list]
        )

        # Load category items
        batch.wildcards.wildcard_list.change(
            fn=load_wildcard_category,
            inputs=[batch.wildcards.wildcard_list],
            outputs=[batch.wildcards.wildcard_items, batch.wildcards.wildcard_preview]
        )

        # Add item
        batch.wildcards.add_item_btn.click(
            fn=add_wildcard_item,
            inputs=[batch.wildcards.wildcard_list, batch.wildcards.wildcard_items, batch.wildcards.add_item_input],
            outputs=[batch.wildcards.wildcard_items, batch.wildcards.wildcard_status]
        )

        # Create category
        batch.wildcards.create_category_btn.click(
            fn=create_wildcard_category,
            inputs=[batch.wildcards.new_category_name],
            outputs=[batch.wildcards.wildcard_list, batch.wildcards.wildcard_status]
        )

        # Delete category
        batch.wildcards.delete_category_btn.click(
            fn=delete_wildcard_category,
            inputs=[batch.wildcards.wildcard_list],
            outputs=[batch.wildcards.wildcard_list, batch.wildcards.wildcard_items, batch.wildcards.wildcard_status]
        )

        # Save wildcards to file
        batch.wildcards.save_wildcards_btn.click(
            fn=lambda cat, items: (save_wildcard_category(cat, items), save_wildcards_to_file())[1],
            inputs=[batch.wildcards.wildcard_list, batch.wildcards.wildcard_items],
            outputs=[batch.wildcards.wildcard_status]
        )

        # Reload wildcards
        batch.wildcards.reload_wildcards_btn.click(
            fn=reload_wildcards,
            inputs=[],
            outputs=[batch.wildcards.wildcard_list, batch.wildcards.wildcard_status]
        )

        # Import wildcards from source files
        batch.wildcards.import_wildcards_btn.click(
            fn=import_wildcards_from_source,
            inputs=[],
            outputs=[batch.wildcards.wildcard_list, batch.wildcards.wildcard_status]
        )

        # Generate wildcards from Ollama LLM
        batch.wildcards.generate_from_ollama_btn.click(
            fn=generate_wildcards_from_ollama,
            inputs=[batch.wildcards.ollama_prompt, batch.wildcards.ollama_count, batch.wildcards.ollama_complexity],
            outputs=[batch.wildcards.wildcard_items, batch.wildcards.wildcard_status]
        )

        # Generate and save as category in one step
        batch.wildcards.generate_and_save_btn.click(
            fn=generate_and_save_wildcard_category,
            inputs=[batch.wildcards.ollama_prompt, batch.wildcards.ollama_count, batch.wildcards.ollama_complexity],
            outputs=[batch.wildcards.wildcard_list, batch.wildcards.wildcard_items, batch.wildcards.wildcard_status]
        )

        # Wire Style Editor
        # Load style
        batch.styles.style_list.change(
            fn=load_style,
            inputs=[batch.styles.style_list],
            outputs=[batch.styles.style_name_input, batch.styles.style_suffix_input, batch.styles.style_preview]
        )

        # Save style
        batch.styles.save_style_btn.click(
            fn=save_style,
            inputs=[batch.styles.style_name_input, batch.styles.style_suffix_input],
            outputs=[batch.styles.style_status]
        )

        # Delete style
        batch.styles.delete_style_btn.click(
            fn=delete_style,
            inputs=[batch.styles.style_name_input],
            outputs=[batch.styles.style_list, batch.styles.style_status]
        )

        # Create style
        batch.styles.create_style_btn.click(
            fn=create_style,
            inputs=[batch.styles.new_style_name, batch.styles.new_style_suffix],
            outputs=[batch.styles.style_list, batch.styles.style_status]
        )

        # Reload styles
        batch.styles.reload_styles_btn.click(
            fn=reload_styles,
            inputs=[],
            outputs=[batch.styles.style_list, batch.styles.style_status]
        )

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
