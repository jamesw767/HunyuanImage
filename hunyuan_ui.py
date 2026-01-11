#!/usr/bin/env python3
"""
HunyuanImage-3.0 Quantized Model UI
A comprehensive Gradio interface for text-to-image generation.
Now with Ollama LLM integration for prompt enhancement and batch processing.
"""

import os
import sys

# Force fully offline mode - no network calls for HuggingFace/transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import random
import json
import shutil
import psutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
import gradio as gr
from PIL import Image

# Global model variable
model = None
model_loaded = False
import threading
model_load_lock = threading.Lock()  # Prevent double-loading race conditions

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

# Batch manager integration
try:
    import batch_manager as bm
    batch_manager_available = True
except ImportError:
    batch_manager_available = False
    print("Warning: batch_manager.py not found - some batch features unavailable")

# Wildcard editor integration
try:
    import wildcard_editor as we
    wildcard_editor_available = True
except ImportError:
    wildcard_editor_available = False
    print("Warning: wildcard_editor.py not found - wildcard editing unavailable")

# LM Studio integration
try:
    import lmstudio_client as lms
    lmstudio_available = True
except ImportError:
    lmstudio_available = False
    print("Warning: lmstudio_client.py not found - LM Studio unavailable")

# Ollama globals
ollama_enhancer = None
ollama_generator = None
# Fallback list only used if Ollama server is not running at startup
OLLAMA_MODELS = ["qwen2.5:7b-instruct"]

# Output directory
OUTPUT_DIR = Path("/media/james/DataDrive/jamesw767/Hun3d/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# History file for tracking generations
HISTORY_FILE = OUTPUT_DIR / "generation_history.json"

# Session directory for regular generation (like batch)
current_session_dir = None
session_counter = 0

def get_or_create_session_dir(prompt: str = "") -> Path:
    """Get or create a session directory for the current generation session.
    Creates directories like: outputs/session_20250104_143022_001_cyberpunk/
    """
    global current_session_dir, session_counter

    if current_session_dir is None or not current_session_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_counter += 1
        # Create a short safe name from the prompt
        safe_name = "".join(c for c in prompt[:25] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        if not safe_name:
            safe_name = "session"
        dir_name = f"session_{timestamp}_{session_counter:03d}_{safe_name}"
        current_session_dir = OUTPUT_DIR / dir_name
        current_session_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SESSION] Created new session directory: {current_session_dir}")

    return current_session_dir

def reset_session_dir():
    """Reset the session directory to force creation of a new one."""
    global current_session_dir
    current_session_dir = None

def log_render_diagnostics(prompt: str, seed: int, image_size: str, steps: int,
                           negative_prompt: str = None, style: str = None,
                           batch_idx: int = 0, batch_total: int = 1,
                           use_ollama: bool = False, ollama_model: str = None,
                           session_type: str = "single", guidance_scale: float = 5.0):
    """Log comprehensive diagnostics at the start of each render.

    This helps debug issues that occur hours into long render sessions.
    """
    print("\n" + "=" * 80)
    print(f"{'='*25} RENDER DIAGNOSTICS {'='*25}")
    print("=" * 80)

    # Timestamp with timezone info
    now = datetime.now()
    print(f"[TIMESTAMP] {now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (local)")
    print(f"[SESSION TYPE] {session_type} | Image {batch_idx + 1}/{batch_total}")

    # GPU Memory Status
    if torch.cuda.is_available():
        try:
            gpu_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            mem_allocated = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(gpu_idx) / (1024**3)
            mem_total = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
            mem_free = mem_total - mem_reserved

            print(f"[GPU] Device {gpu_idx}: {gpu_name}")
            print(f"[GPU MEMORY] Allocated: {mem_allocated:.2f} GB | Reserved: {mem_reserved:.2f} GB | Free: {mem_free:.2f} GB | Total: {mem_total:.2f} GB")

            # Additional CUDA info
            print(f"[CUDA] Version: {torch.version.cuda} | PyTorch: {torch.__version__}")
        except Exception as e:
            print(f"[GPU ERROR] Could not get GPU info: {e}")
    else:
        print("[GPU] CUDA not available!")

    # System Memory Status
    try:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        print(f"[SYSTEM RAM] Used: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}%)")
        print(f"[SWAP] Used: {swap.used / (1024**3):.2f} GB / {swap.total / (1024**3):.2f} GB ({swap.percent}%)")
    except Exception as e:
        print(f"[MEMORY ERROR] Could not get system memory: {e}")

    # Model Status
    global model, model_loaded
    print(f"[MODEL] Loaded: {model_loaded} | Model object exists: {model is not None}")
    if model is not None:
        try:
            # Try to get model info if available
            if hasattr(model, 'device'):
                print(f"[MODEL DEVICE] {model.device}")
            if hasattr(model, 'dtype'):
                print(f"[MODEL DTYPE] {model.dtype}")
        except Exception as e:
            print(f"[MODEL INFO] Could not get detailed model info: {e}")

    # Generation Parameters
    print("-" * 80)
    print(f"[PARAMS] Seed: {seed}")
    print(f"[PARAMS] Image Size: {image_size}")
    print(f"[PARAMS] Inference Steps: {steps}")
    print(f"[PARAMS] Guidance Scale (CFG): {guidance_scale}")
    print(f"[PARAMS] Style: {style if style else 'None'}")
    print(f"[PARAMS] Use Ollama: {use_ollama}" + (f" ({ollama_model})" if use_ollama and ollama_model else ""))

    # Prompt Info
    print("-" * 80)
    print(f"[PROMPT] Length: {len(prompt)} chars")
    print(f"[PROMPT FULL] {prompt}")
    if negative_prompt and negative_prompt.strip():
        print(f"[NEGATIVE] {negative_prompt}")

    # Uptime / Process info
    try:
        process = psutil.Process()
        create_time = datetime.fromtimestamp(process.create_time())
        uptime = now - create_time
        hours, remainder = divmod(uptime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"[PROCESS] Uptime: {int(hours)}h {int(minutes)}m {int(seconds)}s | PID: {process.pid}")
        print(f"[PROCESS MEMORY] RSS: {process.memory_info().rss / (1024**3):.2f} GB")
    except Exception as e:
        print(f"[PROCESS ERROR] Could not get process info: {e}")

    print("=" * 80)
    print("")

# UI Configuration file
UI_CONFIG_FILE = Path(__file__).parent / "ui_config.json"

def load_ui_colors():
    """Load UI colors from configuration file."""
    defaults = {
        "background_main": "#0b0f19",
        "background_secondary": "#111827",
        "background_input": "#1f2937",
        "border_default": "#4b5563",
        "border_accent": "#374151",
        "text_primary": "#f3f4f6",
        "text_secondary": "#e5e7eb",
        "text_muted": "#9ca3af",
        "button_primary": "#2563eb",
        "button_primary_hover": "#1d4ed8",
        "button_secondary": "#374151",
        "button_danger": "#dc2626",
    }

    if UI_CONFIG_FILE.exists():
        try:
            with open(UI_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            colors = config.get("colors", {})
            return {
                "background_main": colors.get("background", {}).get("main", defaults["background_main"]),
                "background_secondary": colors.get("background", {}).get("secondary", defaults["background_secondary"]),
                "background_input": colors.get("background", {}).get("input", defaults["background_input"]),
                "border_default": colors.get("border", {}).get("default", defaults["border_default"]),
                "border_accent": colors.get("border", {}).get("accent", defaults["border_accent"]),
                "text_primary": colors.get("text", {}).get("primary", defaults["text_primary"]),
                "text_secondary": colors.get("text", {}).get("secondary", defaults["text_secondary"]),
                "text_muted": colors.get("text", {}).get("muted", defaults["text_muted"]),
                "button_primary": colors.get("button", {}).get("primary", defaults["button_primary"]),
                "button_primary_hover": colors.get("button", {}).get("primary_hover", defaults["button_primary_hover"]),
                "button_secondary": colors.get("button", {}).get("secondary", defaults["button_secondary"]),
                "button_danger": colors.get("button", {}).get("danger", defaults["button_danger"]),
            }
        except Exception as e:
            print(f"Error loading UI config: {e}, using defaults")
    return defaults


def generate_custom_css():
    """Generate CSS from configuration colors."""
    c = load_ui_colors()
    return f"""
/* Main background */
body, .gradio-container {{
    background-color: {c['background_main']} !important;
}}

/* Inputs */
input, textarea, .gr-input, .gr-box, .gr-form {{
    background-color: {c['background_input']} !important;
    border: 1px solid {c['border_default']} !important;
    color: {c['text_primary']} !important;
}}

.gr-textbox textarea, .gr-textbox input {{
    background-color: {c['background_input']} !important;
    color: {c['text_primary']} !important;
    border: 1px solid {c['border_default']} !important;
}}

::placeholder {{
    color: {c['text_muted']} !important;
    opacity: 1 !important;
}}

/* Dropdowns */
.secondary-wrap, select, .gr-dropdown {{
    background-color: {c['background_input']} !important;
    color: {c['text_primary']} !important;
    border: 1px solid {c['border_default']} !important;
}}

/* Sliders */
input[type="range"] {{
    background-color: {c['border_accent']} !important;
}}

/* Gallery */
.gallery-container, .gr-gallery {{
    background-color: {c['background_secondary']} !important;
    border: 2px solid {c['border_accent']} !important;
    border-radius: 8px !important;
}}

/* Groups and accordions */
.gr-group, .gr-accordion {{
    background-color: {c['background_secondary']} !important;
    border: 1px solid {c['border_accent']} !important;
    border-radius: 8px !important;
}}

/* Primary buttons */
button.primary, .gr-button-primary {{
    background-color: {c['button_primary']} !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
}}
button.primary:hover, .gr-button-primary:hover {{
    background-color: {c['button_primary_hover']} !important;
}}

/* Secondary buttons */
button.secondary, .gr-button-secondary {{
    background-color: {c['button_secondary']} !important;
    color: {c['text_primary']} !important;
    border: 1px solid {c['border_default']} !important;
}}

/* Stop/danger buttons */
button.stop, .gr-button-stop {{
    background-color: {c['button_danger']} !important;
    color: white !important;
}}

/* Labels and text */
label, .gr-label, .gr-markdown {{
    color: {c['text_secondary']} !important;
}}

/* Tabs */
.gr-tab-nav button {{
    background-color: {c['background_input']} !important;
    color: {c['text_muted']} !important;
    border: 1px solid {c['border_accent']} !important;
}}
.gr-tab-nav button.selected {{
    background-color: {c['button_primary']} !important;
    color: white !important;
}}

/* Checkboxes */
.gr-checkbox input[type="checkbox"] {{
    accent-color: {c['button_primary']} !important;
}}

/* Info text */
.gr-info, .info {{
    color: {c['text_muted']} !important;
}}

/* Headers */
h1, h2, h3, h4 {{
    color: {c['text_primary']} !important;
}}

/* Progress bars */
.progress-bar {{
    background-color: {c['button_primary']} !important;
}}
"""

# Generate CSS from config
CUSTOM_CSS = generate_custom_css()

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
    "1280x720",
    "832x480",
]

# Aspect ratio presets
ASPECT_RATIOS = {
    "1:1 (Square)": "1024x1024",
    "16:9 (Landscape)": "1280x768",
    "16:9 (720p Video)": "1280x720",
    "16:9 (480p Video)": "832x480",
    "9:16 (Portrait)": "768x1280",
    "4:3 (Standard)": "1152x896",
    "3:4 (Portrait Standard)": "896x1152",
    "21:9 (Ultrawide)": "1536x640",
    "9:21 (Tall)": "640x1536",
    "Auto (Model decides)": "auto",
}

# Style presets with suffix text - defaults
DEFAULT_STYLE_PRESETS = {
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

# Style presets file path
STYLE_PRESETS_FILE = OUTPUT_DIR / "style_presets.json"


def load_style_presets() -> dict:
    """Load style presets from JSON file, or use defaults if not found."""
    if STYLE_PRESETS_FILE.exists():
        try:
            with open(STYLE_PRESETS_FILE, 'r') as f:
                presets = json.load(f)
            # Ensure "None" always exists
            if "None" not in presets:
                presets["None"] = ""
            return presets
        except Exception as e:
            print(f"[STYLES] Error loading presets: {e}, using defaults")
    return DEFAULT_STYLE_PRESETS.copy()


def save_style_presets(presets: dict) -> str:
    """Save style presets to JSON file."""
    try:
        # Ensure "None" always exists
        if "None" not in presets:
            presets["None"] = ""
        with open(STYLE_PRESETS_FILE, 'w') as f:
            json.dump(presets, f, indent=2)
        return f"Saved {len(presets)} style presets"
    except Exception as e:
        return f"Error saving presets: {e}"


def add_style_preset(name: str, suffix: str) -> tuple:
    """Add or update a style preset."""
    global STYLE_PRESETS
    if not name.strip():
        return "Error: Style name cannot be empty", gr.update()

    name = name.strip()
    # Add comma prefix if not present and suffix is not empty
    if suffix.strip() and not suffix.strip().startswith(','):
        suffix = ", " + suffix.strip()

    STYLE_PRESETS[name] = suffix
    save_style_presets(STYLE_PRESETS)
    choices = list(STYLE_PRESETS.keys())
    return f"Added/updated style: {name}", gr.update(choices=choices, value=name)


def delete_style_preset(name: str) -> tuple:
    """Delete a style preset."""
    global STYLE_PRESETS
    if name == "None":
        return "Cannot delete 'None' style", gr.update()
    if name not in STYLE_PRESETS:
        return f"Style not found: {name}", gr.update()

    del STYLE_PRESETS[name]
    save_style_presets(STYLE_PRESETS)
    choices = list(STYLE_PRESETS.keys())
    return f"Deleted style: {name}", gr.update(choices=choices, value="None")


def get_style_suffix(name: str) -> str:
    """Get the suffix text for a style."""
    return STYLE_PRESETS.get(name, "")


def reset_style_presets() -> tuple:
    """Reset style presets to defaults."""
    global STYLE_PRESETS
    STYLE_PRESETS = DEFAULT_STYLE_PRESETS.copy()
    save_style_presets(STYLE_PRESETS)
    choices = list(STYLE_PRESETS.keys())
    return f"Reset to {len(STYLE_PRESETS)} default styles", gr.update(choices=choices, value="None")


def get_style_choices() -> List[str]:
    """Get list of style preset names."""
    return list(STYLE_PRESETS.keys())


def export_styles_to_text() -> str:
    """Export all styles as formatted text for display/editing."""
    lines = []
    for name, suffix in STYLE_PRESETS.items():
        if name == "None":
            continue
        lines.append(f"{name}: {suffix}")
    return "\n".join(lines)


def import_styles_from_text(text: str) -> tuple:
    """Import styles from text format (one per line: 'Name: suffix')."""
    global STYLE_PRESETS
    if not text.strip():
        return "No text to import", gr.update()

    new_presets = {"None": ""}
    imported = 0
    errors = []

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        parts = line.split(':', 1)
        if len(parts) != 2:
            errors.append(f"Invalid line: {line[:30]}...")
            continue

        name = parts[0].strip()
        suffix = parts[1].strip()

        if not name:
            continue

        # Add comma prefix if needed
        if suffix and not suffix.startswith(','):
            suffix = ", " + suffix

        new_presets[name] = suffix
        imported += 1

    if imported > 0:
        STYLE_PRESETS = new_presets
        save_style_presets(STYLE_PRESETS)
        choices = list(STYLE_PRESETS.keys())
        status = f"Imported {imported} styles"
        if errors:
            status += f" ({len(errors)} errors)"
        return status, gr.update(choices=choices, value="None")
    else:
        return "No valid styles found in text", gr.update()


# Load presets on startup
STYLE_PRESETS = load_style_presets()

# Quality presets
QUALITY_PRESETS = {
    "Draft (Fast)": {"steps": 15, "description": "Quick preview, lower quality"},
    "Standard": {"steps": 20, "description": "Good balance of speed and quality"},
    "High Quality": {"steps": 30, "description": "Better details, slower"},
    "Maximum": {"steps": 50, "description": "Best quality, slowest"},
}


def load_model():
    """Load the quantized HunyuanImage-3.0 model."""
    global model, model_loaded, model_load_lock

    # Quick check without lock
    if model_loaded and model is not None:
        yield "Model already loaded and ready!"
        return

    # Try to acquire lock (non-blocking to give feedback)
    if not model_load_lock.acquire(blocking=False):
        yield "Model is currently being loaded by another request..."
        # Wait for the other load to complete
        model_load_lock.acquire()
        model_load_lock.release()
        if model_loaded:
            yield "Model loaded by another request. Ready!"
        return

    try:
        # Double-check after acquiring lock
        if model_loaded and model is not None:
            yield "Model already loaded and ready!"
            return

        # Check if GPU already has significant memory usage (model might be loaded but flag not set)
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            if mem_used > 30:  # More than 30GB suggests model is already loaded
                yield f"WARNING: GPU already has {mem_used:.1f}GB allocated. Model may already be loaded."
                yield "If you want to reload, unload first."
                return

        print("[LOAD] Step 1: Importing transformers...")
        yield "Step 1: Importing libraries..."
        from transformers import AutoModelForCausalLM

        print("[LOAD] Step 2: Importing SDNQ...")
        from sdnq import SDNQConfig  # Registers SDNQ into transformers

        model_id = "/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage3-SDNQ"

        print(f"[LOAD] Step 3: Loading model from {model_id}...")
        yield "Step 2: Loading quantized HunyuanImage-3.0 model on GPU... (this may take 1-2 minutes)"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # Load to the Blackwell GPU
            moe_impl="eager",
            local_files_only=True,  # Fully offline - no network calls
        )

        print("[LOAD] Step 4: Model loaded successfully!")
        yield "Step 3: Model weights loaded, setting up tokenizer..."

        # Model is loaded - set flag FIRST in case tokenizer fails
        model_loaded = True

        # Try to load tokenizer (may fail with local_files_only but model still works)
        print("[LOAD] Step 5: Loading tokenizer...")
        try:
            model.load_tokenizer(model_id, local_files_only=True)
            print("[LOAD] Step 5: Tokenizer loaded successfully!")
        except Exception as tok_err:
            print(f"[LOAD] Tokenizer load warning (model still works): {tok_err}")
            # Try without local_files_only as fallback
            try:
                model.load_tokenizer(model_id)
                print("[LOAD] Step 5: Tokenizer loaded (fallback method)")
            except Exception:
                print("[LOAD] Tokenizer fallback also failed - model may still work for generation")

        # Show GPU memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            yield f"Model loaded! Using {mem_used:.1f}GB / {mem_total:.1f}GB GPU memory. Ready to generate."
        else:
            yield "Model loaded successfully! Ready to generate images."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[LOAD ERROR] {str(e)}")
        print(f"[LOAD TRACEBACK]\n{error_details}")
        yield f"Error loading model: {str(e)}\n\nDetails:\n{error_details}"
    finally:
        model_load_lock.release()


def unload_model():
    """Unload the image generation model to free GPU memory for LLM usage."""
    global model, model_loaded

    if not model_loaded or model is None:
        return "Model not loaded - nothing to unload"

    try:
        # Delete model and clear CUDA cache
        del model
        model = None
        model_loaded = False

        # Clear CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Report freed memory
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            return f"Model unloaded! GPU memory freed. Current usage: {mem_used:.1f}GB"

        return "Model unloaded successfully!"

    except Exception as e:
        return f"Error unloading model: {str(e)}"


def get_model_status():
    """Get current model loading status and GPU memory info."""
    global model, model_loaded

    status_lines = []

    if model_loaded and model is not None:
        status_lines.append("**Image Model: LOADED**")
    else:
        status_lines.append("**Image Model: NOT LOADED**")

    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            status_lines.append(f"GPU: {gpu_name}")
            status_lines.append(f"Memory: {mem_used:.1f}GB used / {mem_total:.1f}GB total")

            if not model_loaded:
                status_lines.append("")
                status_lines.append("*GPU is free for LLM prompt generation*")
        except Exception as e:
            status_lines.append(f"GPU info error: {e}")
    else:
        status_lines.append("No CUDA GPU available")

    return "\n".join(status_lines)


# Shot/camera keywords to strip from prompts before applying shot styles
SHOT_KEYWORDS = [
    # Shot types
    "extreme close-up", "extreme closeup", "ECU",
    "close-up", "closeup", "close up", "CU",
    "medium close-up", "medium closeup", "MCU",
    "medium shot", "mid-shot", "mid shot", "MS",
    "medium wide", "medium wide shot", "MWS", "cowboy shot",
    "wide shot", "wide-shot", "full-body shot", "full body shot", "long shot", "WS",
    "extreme wide", "extreme wide shot", "EWS", "establishing shot",
    # Camera angles
    "eye-level", "eye level",
    "high-angle", "high angle", "looking down",
    "low-angle", "low angle", "looking up",
    "dutch angle", "tilted frame", "canted angle",
    "bird's eye", "birds eye", "top-down", "overhead", "aerial",
    "worm's eye", "worms eye", "ground level",
    "over-the-shoulder", "over the shoulder", "OTS",
    "POV shot", "first-person", "first person",
    # Camera movement
    "tracking shot", "dolly shot", "crane shot", "steadicam", "handheld",
]

def strip_shot_keywords(prompt: str) -> str:
    """Remove shot/camera direction keywords from prompt to allow style-based shots."""
    result = prompt
    for keyword in SHOT_KEYWORDS:
        # Case-insensitive removal with comma/semicolon cleanup
        import re
        pattern = re.compile(r',?\s*' + re.escape(keyword) + r'\s*,?', re.IGNORECASE)
        result = pattern.sub(', ', result)
    # Clean up multiple commas/semicolons and trailing punctuation
    result = re.sub(r'[,;]\s*[,;]+', ', ', result)
    result = re.sub(r'\s+', ' ', result)
    result = result.strip(' ,;')
    return result


def apply_style(prompt: str, style: str, strip_shots: bool = True) -> str:
    """Apply a style preset to the prompt.

    Args:
        prompt: The base prompt
        style: Style name from STYLE_PRESETS
        strip_shots: If True and style is a shot/camera style, remove existing shot keywords
    """
    style_suffix = STYLE_PRESETS.get(style, "")

    # If applying a shot-related style, strip existing shot keywords first
    if strip_shots and style_suffix:
        # Check if this is a shot/camera style (contains shot-related terms)
        shot_terms = ["shot", "angle", "eye", "close", "wide", "medium", "POV", "tracking", "dolly", "crane"]
        is_shot_style = any(term.lower() in style.lower() or term.lower() in style_suffix.lower() for term in shot_terms)
        if is_shot_style:
            prompt = strip_shot_keywords(prompt)

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


def refresh_ollama_status_and_models():
    """Check status and refresh all Ollama model dropdowns"""
    status = check_ollama_status()
    models = get_ollama_models_list()
    # Return status + 3 dropdown updates (ollama_model, batch_ollama_model, wc_llm_model)
    return (
        status,
        gr.update(choices=models, value=models[0] if models else None),
        gr.update(choices=models, value=models[0] if models else None),
        gr.update(choices=models, value=models[0] if models else None)
    )


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
    """Start the Ollama server and refresh all model dropdowns"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available", gr.update(), gr.update(), gr.update()
    success, msg = ollama_manager.start()
    models = get_ollama_models_list()
    dropdown_update = gr.update(choices=models, value=models[0] if models else None)
    return msg, dropdown_update, dropdown_update, dropdown_update


def stop_ollama_server():
    """Stop the Ollama server"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available"
    success, msg = ollama_manager.stop()
    return msg


def pull_ollama_model(model_name: str):
    """Pull/install a new Ollama model"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available", gr.update(), gr.update(), gr.update()

    if not model_name or not model_name.strip():
        return "Please enter a model name", gr.update(), gr.update(), gr.update()

    # Start server if not running
    if not ollama_manager.is_running():
        ollama_manager.start()

    yield f"Pulling {model_name}... (this may take a while)", gr.update(), gr.update(), gr.update()

    def progress_cb(status, pct):
        pass  # Gradio doesn't support real-time updates well in this context

    success, msg = ollama_manager.pull_model(model_name.strip())
    models = get_ollama_models_list()
    new_value = model_name.strip() if success else (models[0] if models else None)
    dropdown_update = gr.update(choices=models, value=new_value)
    yield msg, dropdown_update, dropdown_update, dropdown_update


def delete_ollama_model(model_name: str):
    """Delete an Ollama model"""
    if not ollama_available or not ollama_manager:
        return "Ollama not available", gr.update(), gr.update(), gr.update()

    if not model_name:
        return "No model selected", gr.update(), gr.update(), gr.update()

    success, msg = ollama_manager.delete_model(model_name)
    models = get_ollama_models_list()
    dropdown_update = gr.update(choices=models, value=models[0] if models else None)
    return msg, dropdown_update, dropdown_update, dropdown_update


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
                           styles_selected: List[str], llm_backend: str,
                           ollama_model: str, enhance_prompts: bool,
                           ollama_length: str = "medium",
                           ollama_complexity: str = "detailed") -> List[dict]:
    """Generate all prompt combinations for batch"""
    global ollama_generator, ollama_enhancer

    themes = [t.strip() for t in themes_text.strip().split('\n') if t.strip()]
    if not themes:
        return []

    all_prompts = []

    # Check which backend to use
    use_lmstudio = (llm_backend == "LM Studio" and lmstudio_available and lms.check_lmstudio_available())
    use_ollama = (llm_backend == "Ollama" and ollama_available)

    print(f"[DEBUG] LLM Backend: {llm_backend}, use_lmstudio={use_lmstudio}, use_ollama={use_ollama}")

    # Initialize Ollama if needed (and using Ollama backend)
    if use_ollama and (variations_per_theme > 1 or enhance_prompts):
        if ollama_generator is None:
            ollama_generator = PromptGenerator(model=ollama_model)
        if ollama_enhancer is None:
            ollama_enhancer = PromptEnhancer(model=ollama_model)

    # Initialize LM Studio client if needed
    lmstudio_client = None
    if use_lmstudio and (variations_per_theme > 1 or enhance_prompts):
        lmstudio_client = lms.get_client()

    for theme in themes:
        print(f"\n{'='*60}")
        print(f"[DEBUG] Processing theme: {theme}")
        print(f"[DEBUG] variations_per_theme={variations_per_theme}, use_ollama={use_ollama}, use_lmstudio={use_lmstudio}")
        print(f"[DEBUG] wildcard_available={wildcard_available}, has_wildcards={wildcard_manager.has_wildcards(theme) if wildcard_manager else 'N/A'}")
        print(f"{'='*60}")

        # Generate variations using LLM (Ollama or LM Studio)
        if variations_per_theme > 1 and ((use_ollama and ollama_generator) or (use_lmstudio and lmstudio_client)):
            print(f"[DEBUG] Taking LLM variation path (variations > 1, backend={llm_backend})")
            variations = []
            for var_idx in range(variations_per_theme):
                try:
                    # Expand wildcards FRESH for each variation BEFORE sending to LLM
                    # This ensures each variation gets different random wildcard values
                    expanded_theme = theme
                    if wildcard_available and wildcard_manager and wildcard_manager.has_wildcards(theme):
                        expanded_theme = wildcard_manager.process_prompt(theme)
                        print(f"[DEBUG] Variation {var_idx+1}: Wildcards expanded: '{theme}' -> '{expanded_theme}'")
                    else:
                        print(f"[DEBUG] Variation {var_idx+1}: No wildcards to expand, using: '{expanded_theme}'")

                    # Now generate ONE variation from the expanded theme
                    if use_lmstudio and lmstudio_client:
                        # Use LM Studio
                        from ollama_prompts import get_generation_system_prompt
                        system_prompt = get_generation_system_prompt(ollama_length, ollama_complexity)
                        response = lmstudio_client.generate(
                            prompt=f"Generate one unique creative image prompt based on: {expanded_theme}",
                            system=system_prompt,
                            temperature=0.85,
                            max_tokens=256
                        )
                        var_prompt = response.text.strip()
                        if var_prompt:
                            print(f"[DEBUG] Variation {var_idx+1}: LM Studio returned: '{var_prompt[:80]}...'")
                            variations.append(var_prompt)
                        else:
                            print(f"[DEBUG] Variation {var_idx+1}: LM Studio returned empty, using expanded theme")
                            variations.append(expanded_theme)
                    else:
                        # Use Ollama
                        var_prompts = ollama_generator.generate_themed_prompts(
                            expanded_theme, count=1, temperature=0.85,
                            length=ollama_length, complexity=ollama_complexity
                        )
                        if var_prompts:
                            print(f"[DEBUG] Variation {var_idx+1}: Ollama returned: '{var_prompts[0][:80]}...'")
                            variations.append(var_prompts[0])
                        else:
                            print(f"[DEBUG] Variation {var_idx+1}: Ollama returned empty, using expanded theme")
                            variations.append(expanded_theme)
                except Exception as e:
                    print(f"[DEBUG] Variation {var_idx+1}: Exception: {e}")
                    # Fall back to just the expanded theme
                    expanded_theme = theme
                    if wildcard_available and wildcard_manager and wildcard_manager.has_wildcards(theme):
                        expanded_theme = wildcard_manager.process_prompt(theme)
                    variations.append(expanded_theme)

            if not variations:
                variations = [theme]
        else:
            # No LLM - still loop to get unique wildcard values for each variation
            # "[color] bird" becomes "red bird", then "blue bird", then "green bird"
            print(f"[DEBUG] Taking non-LLM path")
            variations = []
            for var_idx in range(variations_per_theme):
                expanded_theme = theme
                if wildcard_available and wildcard_manager and wildcard_manager.has_wildcards(theme):
                    expanded_theme = wildcard_manager.process_prompt(theme)
                    print(f"[DEBUG] Variation {var_idx+1}: Wildcards expanded: '{theme}' -> '{expanded_theme}'")
                else:
                    print(f"[DEBUG] Variation {var_idx+1}: No wildcards, using: '{expanded_theme}'")
                variations.append(expanded_theme)

        print(f"[DEBUG] Total variations created: {len(variations)}")

        # Enhance each variation if requested
        if enhance_prompts and ((use_ollama and ollama_enhancer) or (use_lmstudio and lmstudio_client)):
            enhanced_variations = []
            for v in variations:
                try:
                    if use_lmstudio and lmstudio_client:
                        # Use LM Studio for enhancement
                        from ollama_prompts import get_enhance_system_prompt
                        system_prompt = get_enhance_system_prompt(ollama_length, ollama_complexity)
                        response = lmstudio_client.generate(
                            prompt=v,
                            system=system_prompt,
                            temperature=0.7,
                            max_tokens=512
                        )
                        enhanced = response.text.strip() if response.text.strip() else v
                    else:
                        # Use Ollama for enhancement
                        enhanced = ollama_enhancer.enhance(
                            v, temperature=0.7,
                            length=ollama_length, complexity=ollama_complexity
                        )
                    enhanced_variations.append(enhanced)
                except:
                    enhanced_variations.append(v)
            variations = enhanced_variations

        # Create combinations with styles
        # Each variation gets ONE seed that's shared across all its style variants
        # This ensures same prompt + same seed + different shots = consistent subject
        styles = styles_selected if styles_selected else ["None"]
        print(f"[DEBUG] Styles to apply: {styles}")
        for var_idx, variation in enumerate(variations):
            # Generate ONE seed per variation - all styles of this variation share it
            variation_seed = random.randint(0, 2**32 - 1)
            print(f"[DEBUG] Variation {var_idx+1} seed: {variation_seed} (shared across {len(styles)} styles)")
            for style in styles:
                all_prompts.append({
                    "prompt": variation,
                    "style": style,
                    "original_theme": theme,
                    "variation_seed": variation_seed,  # Same seed for all styles of this variation
                    "variation_index": var_idx,
                })

    # Debug: show first few prompts
    print(f"\n[DEBUG] Total prompts created: {len(all_prompts)}")
    for i, p in enumerate(all_prompts[:6]):
        print(f"[DEBUG] Prompt {i+1}: style='{p['style']}', prompt='{p['prompt'][:60]}...'")
    if len(all_prompts) > 6:
        print(f"[DEBUG] ... and {len(all_prompts) - 6} more")

    return all_prompts


def run_batch_generation(
    themes_text: str,
    variations_per_theme: int,
    styles_selected: List[str],
    images_per_combo: int,
    llm_backend: str,
    ollama_model: str,
    enhance_prompts: bool,
    aspect_ratio: str,
    quality_preset: str,
    random_seeds: bool,
    batch_name: str,
    ollama_length: str = "medium",
    ollama_complexity: str = "detailed",
    negative_prompt: str = "",
    guidance_scale: float = 5.0,
    progress=gr.Progress()
):
    """Run the batch generation process"""
    global model, model_loaded, batch_running, batch_stop_requested, batch_results

    # Debug: Print received parameters
    print(f"\n{'='*60}")
    print(f"[BATCH START] Received parameters:")
    print(f"  llm_backend = '{llm_backend}'")
    print(f"  aspect_ratio = '{aspect_ratio}'")
    print(f"  ASPECT_RATIOS.get('{aspect_ratio}') = '{ASPECT_RATIOS.get(aspect_ratio, 'NOT FOUND')}'")
    print(f"  quality_preset = '{quality_preset}'")
    print(f"  Available aspect ratios: {list(ASPECT_RATIOS.keys())}")
    print(f"{'='*60}\n")

    # Debug: Check model state
    print(f"[BATCH CHECK] model_loaded={model_loaded}, model is None={model is None}, model type={type(model)}")

    # Check if model exists (flag might not be set due to tokenizer error)
    if model is None:
        yield [], "Please load the model first!", "", []
        return

    # Fix flag if model exists but flag not set
    if not model_loaded and model is not None:
        print("[BATCH] Model exists but flag was False - fixing flag")
        model_loaded = True

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

        progress(0, desc=f"Generating prompts with {llm_backend}...")
        yield [], f"Generating prompt variations with {llm_backend} ({ollama_length}/{ollama_complexity})...", "", []

        # Generate all prompt combinations
        prompts = generate_batch_prompts(
            themes_text, variations_per_theme, styles_selected,
            llm_backend, ollama_model, enhance_prompts, ollama_length, ollama_complexity
        )

        if not prompts:
            yield [], "No prompts generated. Please enter at least one theme.", "", []
            batch_running = False
            return

        # Expand for images_per_combo with tracking for interleaving
        all_jobs = []
        for prompt_idx, p in enumerate(prompts):
            for combo_idx in range(images_per_combo):
                all_jobs.append({
                    **p,
                    "combo_index": combo_idx,
                    "prompt_index": prompt_idx,
                })

        # Always interleave in batch mode for variety
        # Instead of: theme1_v1, theme1_v2, theme1_v3, theme2_v1...
        # We get:     theme1_v1, theme2_v1, theme3_v1, theme1_v2...
        if len(all_jobs) > 1:
            # Log order before sorting
            print(f"[BATCH] Before interleave - first 6 jobs:")
            for i, j in enumerate(all_jobs[:6]):
                print(f"  [{i}] prompt_idx={j['prompt_index']}, combo_idx={j['combo_index']}, prompt='{j['prompt'][:40]}...'")

            all_jobs.sort(key=lambda x: (x["combo_index"], x["prompt_index"]))

            # Log order after sorting
            print(f"\n[BATCH] After interleave - first 12 jobs (should alternate prompts):")
            for i, j in enumerate(all_jobs[:12]):
                print(f"  [{i}] prompt_idx={j['prompt_index']}, combo_idx={j['combo_index']}, prompt='{j['prompt'][:40]}...'")
            print(f"[BATCH] Interleaved {len(all_jobs)} jobs to alternate between prompts")

        total_jobs = len(all_jobs)

        # Get settings - with debug
        image_size = ASPECT_RATIOS.get(aspect_ratio, "1024x1024")
        print(f"[BATCH] Using image_size = '{image_size}' for aspect_ratio = '{aspect_ratio}'")
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
                "negative_prompt": negative_prompt,
                "ollama_model": ollama_model,
                "enhance_prompts": enhance_prompts
            },
            "prompts": prompts
        }
        with open(batch_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save prompt run to separate file for reuse/editing
        prompt_run_settings = {
            "themes": themes_text.split('\n'),
            "variations_per_theme": variations_per_theme,
            "styles": styles_selected,
            "images_per_combo": images_per_combo,
            "aspect_ratio": aspect_ratio,
            "quality": quality_preset,
            "negative_prompt": negative_prompt,
            "ollama_model": ollama_model,
            "ollama_length": ollama_length,
            "ollama_complexity": ollama_complexity,
            "enhance_prompts": enhance_prompts,
            "random_seeds": random_seeds
        }
        prompt_run_path = save_prompt_run(batch_name, prompts, prompt_run_settings)
        yield [], f"Saved {len(prompts)} prompts to: {Path(prompt_run_path).name}\nStarting rendering...", f"Output: {batch_dir}", []

        generated_images = []

        for idx, job in enumerate(all_jobs):
            if batch_stop_requested:
                yield generated_images, f"Batch stopped at {idx}/{total_jobs}", f"Output: {batch_dir}", generated_images[-12:] if generated_images else []
                break

            prompt = job["prompt"]
            style = job["style"]

            # Use variation_seed if available (ensures same seed for all shots of same scene)
            # This is key for image-to-video workflows where you want consistent subjects
            if "variation_seed" in job:
                seed = job["variation_seed"]
                print(f"[BATCH] Using variation_seed={seed} for consistent subject across shots")
            elif random_seeds:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = (idx * 12345) % (2**32)

            # Process wildcards if present - fresh random for each image
            original_prompt = prompt
            if wildcard_available and wildcard_manager and wildcard_manager.has_wildcards(prompt):
                prompt = wildcard_manager.process_prompt(prompt)

            # Apply style
            styled_prompt = apply_style(prompt, style)

            # Add negative prompt if provided
            full_prompt = styled_prompt
            if negative_prompt and negative_prompt.strip():
                full_prompt = f"{styled_prompt}  {negative_prompt.strip()}"

            progress((idx + 1) / total_jobs, desc=f"Generating {idx + 1}/{total_jobs}...")
            yield generated_images, f"Generating {idx + 1}/{total_jobs}: {prompt[:50]}...", f"Output: {batch_dir}", generated_images[-12:] if generated_images else []

            try:
                start_time = time.time()

                # Log comprehensive diagnostics before each render
                log_render_diagnostics(
                    prompt=full_prompt,
                    seed=seed,
                    image_size=image_size,
                    steps=steps,
                    negative_prompt=negative_prompt,
                    style=style,
                    batch_idx=idx,
                    batch_total=total_jobs,
                    use_ollama=enhance_prompts,
                    ollama_model=ollama_model if enhance_prompts else None,
                    session_type="batch",
                    guidance_scale=guidance_scale
                )

                # Check if tokenizer is still loaded (can get lost due to memory pressure)
                if not hasattr(model, '_tkwrapper') or model._tkwrapper is None:
                    print("[BATCH] WARNING: Tokenizer missing, attempting to reload...")
                    model_id = "/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage3-SDNQ"
                    try:
                        model.load_tokenizer(model_id)
                        print("[BATCH] Tokenizer reloaded successfully!")
                    except Exception as tok_err:
                        print(f"[BATCH] Tokenizer reload failed: {tok_err}")
                        raise RuntimeError(f"Tokenizer not loaded and reload failed: {tok_err}")

                image = model.generate_image(
                    prompt=full_prompt,
                    seed=seed,
                    image_size=image_size,
                    diff_infer_steps=steps,
                    diff_guidance_scale=guidance_scale,
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
                    "negative_prompt": negative_prompt if negative_prompt and negative_prompt.strip() else None,
                    "style": style,
                    "seed": seed,
                    "image_size": image_size,
                    "aspect_ratio": aspect_ratio,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
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

                # Clean up GPU memory after error (helps with OOM recovery)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"[MEMORY] Cleared cache after error")

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
# SINGLE PROMPT CONFIG MANAGEMENT
# ============================================================================

PROMPT_CONFIGS_DIR = OUTPUT_DIR / "prompt_configs"
PROMPT_CONFIGS_DIR.mkdir(exist_ok=True)

# Stop flag for single generation
single_generation_stop = False


def stop_single_generation():
    """Request stop for single generation batch"""
    global single_generation_stop
    single_generation_stop = True
    return "Stop requested... will stop after current image"


def reset_single_generation_stop():
    """Reset the stop flag"""
    global single_generation_stop
    single_generation_stop = False


def save_prompt_config(
    config_name: str,
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
    use_ollama: bool,
    ollama_model: str,
    ollama_length: str,
    ollama_complexity: str,
    batch_count: int,
    overwrite: bool = False
) -> tuple:
    """Save prompt configuration to a JSON file"""
    if not config_name.strip():
        return "Please enter a config name", gr.update()

    config = {
        "config_name": config_name,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "style": style,
        "aspect_ratio": aspect_ratio,
        "custom_size": custom_size,
        "quality_preset": quality_preset,
        "custom_steps": custom_steps,
        "use_custom_steps": use_custom_steps,
        "seed": seed,
        "use_random_seed": use_random_seed,
        "use_ollama": use_ollama,
        "ollama_model": ollama_model,
        "ollama_length": ollama_length,
        "ollama_complexity": ollama_complexity,
        "batch_count": batch_count,
        "saved_at": datetime.now().isoformat()
    }

    safe_name = "".join(c for c in config_name[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")

    if overwrite:
        # Find and overwrite existing file with this name prefix
        existing = list(PROMPT_CONFIGS_DIR.glob(f"{safe_name}_*.json"))
        if existing:
            filepath = existing[0]  # Overwrite the first match
        else:
            filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = PROMPT_CONFIGS_DIR / filename
    else:
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = PROMPT_CONFIGS_DIR / filename

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    return f"Saved: {filepath.name}", gr.update(choices=get_saved_prompt_configs())


def get_saved_prompt_configs() -> List[str]:
    """Get list of saved prompt config files"""
    if not PROMPT_CONFIGS_DIR.exists():
        return []
    configs = sorted(PROMPT_CONFIGS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    return [f.name for f in configs]


def load_prompt_config(config_name: str):
    """Load a saved prompt configuration"""
    if not config_name:
        return [gr.update()] * 16 + ["No config selected"]

    filepath = PROMPT_CONFIGS_DIR / config_name
    if not filepath.exists():
        return [gr.update()] * 16 + [f"Config not found: {config_name}"]

    try:
        with open(filepath, 'r') as f:
            config = json.load(f)

        return [
            gr.update(value=config.get('config_name', '')),           # config_name_input
            gr.update(value=config.get('prompt', '')),                 # prompt
            gr.update(value=config.get('negative_prompt', '')),        # negative_prompt
            gr.update(value=config.get('style', 'None')),              # style
            gr.update(value=config.get('aspect_ratio', '1:1 (Square)')), # aspect_ratio
            gr.update(value=config.get('custom_size', '')),            # custom_size
            gr.update(value=config.get('quality_preset', 'Standard')), # quality_preset
            gr.update(value=config.get('custom_steps', 20)),           # custom_steps
            gr.update(value=config.get('use_custom_steps', False)),    # use_custom_steps
            gr.update(value=config.get('seed', 0)),                    # seed
            gr.update(value=config.get('use_random_seed', True)),      # use_random_seed
            gr.update(value=config.get('use_ollama', False)),          # use_ollama
            gr.update(value=config.get('ollama_model', 'qwen2.5:7b-instruct')), # ollama_model
            gr.update(value=config.get('ollama_length', 'medium')),    # ollama_length
            gr.update(value=config.get('ollama_complexity', 'detailed')), # ollama_complexity
            gr.update(value=config.get('batch_count', 1)),             # batch_count
            f"Loaded: {config_name}\nSaved: {config.get('saved_at', 'unknown')[:19]}"
        ]
    except Exception as e:
        return [gr.update()] * 16 + [f"Error loading: {e}"]


def delete_prompt_config(config_name: str) -> tuple:
    """Delete a saved prompt configuration"""
    if not config_name:
        return "No config selected", gr.update()

    filepath = PROMPT_CONFIGS_DIR / config_name
    if filepath.exists():
        filepath.unlink()
        return f"Deleted: {config_name}", gr.update(choices=get_saved_prompt_configs())
    return f"Not found: {config_name}", gr.update()


# ============================================================================
# PROMPT RUNS (Generated prompt lists for batch rendering)
# ============================================================================

PROMPT_RUNS_DIR = OUTPUT_DIR / "prompt_runs"
PROMPT_RUNS_DIR.mkdir(exist_ok=True)


def save_prompt_run(
    run_name: str,
    prompts: List[dict],
    settings: dict
) -> str:
    """
    Save generated prompts to a JSON file for later use.

    Args:
        run_name: Name for this prompt run
        prompts: List of prompt dicts from generate_batch_prompts()
        settings: Batch settings dict

    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in run_name[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    filename = f"{safe_name}_{timestamp}.json"
    filepath = PROMPT_RUNS_DIR / filename

    run_data = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "total_prompts": len(prompts),
        "settings": settings,
        "prompts": prompts
    }

    with open(filepath, 'w') as f:
        json.dump(run_data, f, indent=2)

    print(f"[PROMPT RUN] Saved {len(prompts)} prompts to: {filepath}")
    return str(filepath)


def get_saved_prompt_runs() -> List[str]:
    """Get list of saved prompt run files"""
    if not PROMPT_RUNS_DIR.exists():
        return []
    runs = sorted(PROMPT_RUNS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    return [f.name for f in runs]


def load_prompt_run(run_name: str) -> tuple:
    """
    Load a saved prompt run.

    Returns:
        (prompts_list, settings_dict, status_message) or (None, None, error_message)
    """
    if not run_name:
        return None, None, "No run selected"

    filepath = PROMPT_RUNS_DIR / run_name
    if not filepath.exists():
        return None, None, f"Run not found: {run_name}"

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        prompts = data.get("prompts", [])
        settings = data.get("settings", {})
        created = data.get("created_at", "unknown")[:19]

        return prompts, settings, f"Loaded: {run_name}\n{len(prompts)} prompts | Created: {created}"
    except Exception as e:
        return None, None, f"Error loading: {e}"


def delete_prompt_run(run_name: str) -> tuple:
    """Delete a saved prompt run"""
    if not run_name:
        return "No run selected", gr.update()

    filepath = PROMPT_RUNS_DIR / run_name
    if filepath.exists():
        filepath.unlink()
        return f"Deleted: {run_name}", gr.update(choices=get_saved_prompt_runs())
    return f"Not found: {run_name}", gr.update()


def preview_prompt_run(run_name: str) -> str:
    """Get a preview of prompts in a saved run"""
    if not run_name:
        return "Select a prompt run to preview"

    filepath = PROMPT_RUNS_DIR / run_name
    if not filepath.exists():
        return f"Run not found: {run_name}"

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        prompts = data.get("prompts", [])
        settings = data.get("settings", {})

        preview_lines = [
            f"**{data.get('run_name', run_name)}**",
            f"Created: {data.get('created_at', 'unknown')[:19]}",
            f"Total prompts: {len(prompts)}",
            "",
            f"Settings: {settings.get('aspect_ratio', 'N/A')} | {settings.get('quality', 'N/A')} | {settings.get('images_per_combo', 1)} img/prompt",
            "",
            "**First 10 prompts:**",
        ]

        for i, p in enumerate(prompts[:10]):
            prompt_text = p.get('prompt', str(p))[:80]
            style = p.get('style', 'None')
            preview_lines.append(f"{i+1}. [{style}] {prompt_text}...")

        if len(prompts) > 10:
            preview_lines.append(f"... and {len(prompts) - 10} more")

        return "\n".join(preview_lines)
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# COMBINE JSON CONFIGS INTO BATCH
# ============================================================================

def load_configs_from_files(files) -> tuple:
    """
    Load multiple JSON config files and extract prompt data.

    Args:
        files: List of uploaded file objects or file paths

    Returns:
        (list of config dicts, status message)
    """
    if not files:
        return [], "No files selected"

    configs = []
    errors = []

    for f in files:
        try:
            # Handle both file objects and paths
            if hasattr(f, 'name'):
                filepath = f.name
            else:
                filepath = str(f)

            with open(filepath, 'r') as fp:
                config = json.load(fp)

            # Validate it has the required fields
            if 'prompt' not in config:
                errors.append(f"{Path(filepath).name}: missing 'prompt' field")
                continue

            configs.append({
                'config': config,
                'source_file': Path(filepath).name
            })
        except json.JSONDecodeError as e:
            errors.append(f"{Path(filepath).name}: invalid JSON - {e}")
        except Exception as e:
            errors.append(f"{Path(filepath).name}: {e}")

    status_parts = [f"Loaded {len(configs)} configs"]
    if errors:
        status_parts.append(f"{len(errors)} errors:")
        status_parts.extend(f"  - {e}" for e in errors[:5])
        if len(errors) > 5:
            status_parts.append(f"  ... and {len(errors) - 5} more")

    return configs, "\n".join(status_parts)


def preview_combined_configs(files) -> str:
    """Preview the configs that will be combined"""
    configs, status = load_configs_from_files(files)

    if not configs:
        return f"No valid configs loaded.\n{status}"

    preview_lines = [
        f"## {len(configs)} Configs to Combine\n",
        status,
        "\n---\n"
    ]

    for i, item in enumerate(configs[:20]):
        config = item['config']
        prompt = config.get('prompt', 'No prompt')[:60]
        style = config.get('style', 'None')
        seed = config.get('seed', 'N/A')
        source = item['source_file']

        preview_lines.append(f"**{i+1}.** `{source}`")
        preview_lines.append(f"   - Prompt: {prompt}...")
        preview_lines.append(f"   - Style: {style} | Seed: {seed}")
        preview_lines.append("")

    if len(configs) > 20:
        preview_lines.append(f"\n*... and {len(configs) - 20} more configs*")

    return "\n".join(preview_lines)


def combine_configs_to_prompt_run(
    files,
    run_name: str,
    images_per_prompt: int = 1,
    use_original_seeds: bool = False
) -> tuple:
    """
    Combine multiple JSON configs into a single prompt run file.

    Args:
        files: List of uploaded config files
        run_name: Name for the new prompt run
        images_per_prompt: How many images to generate per prompt
        use_original_seeds: If True, store original seeds (otherwise generate fresh)

    Returns:
        (success: bool, message: str, dropdown_update)
    """
    configs, load_status = load_configs_from_files(files)

    if not configs:
        return False, f"No valid configs to combine.\n{load_status}", gr.update()

    if not run_name.strip():
        run_name = "combined_configs"

    # Extract prompts in the format expected by prompt runs
    prompts = []
    for item in configs:
        config = item['config']
        prompt_entry = {
            "prompt": config.get('prompt', ''),
            "style": config.get('style', 'None'),
        }
        # Optionally include original seed for reference
        if use_original_seeds:
            prompt_entry["original_seed"] = config.get('seed')
        prompts.append(prompt_entry)

    # Get common settings from the first config (can be overridden when running)
    first_config = configs[0]['config']
    settings = {
        "aspect_ratio": first_config.get('aspect_ratio', '1:1 (Square)'),
        "quality": first_config.get('quality_preset', 'Standard'),
        "negative_prompt": first_config.get('negative_prompt', ''),
        "images_per_combo": images_per_prompt,
        "guidance_scale": first_config.get('guidance_scale', 5.0),
        "random_seeds": not use_original_seeds,
        "source": "combined_configs",
        "source_files": [item['source_file'] for item in configs]
    }

    # Save as prompt run
    try:
        filepath = save_prompt_run(run_name, prompts, settings)
        filename = Path(filepath).name
        return (
            True,
            f"Created prompt run: {filename}\n{len(prompts)} prompts from {len(configs)} configs\nImages per prompt: {images_per_prompt}",
            gr.update(choices=get_saved_prompt_runs(), value=filename)
        )
    except Exception as e:
        return False, f"Error saving prompt run: {e}", gr.update()


def scan_output_dirs_for_configs() -> List[str]:
    """Scan output directories for JSON config files"""
    config_files = []

    # Scan main outputs directory
    for json_file in OUTPUT_DIR.glob("**/*.json"):
        # Skip manifest files and prompt runs
        if json_file.name in ['manifest.json', 'results.json']:
            continue
        if 'prompt_runs' in str(json_file):
            continue
        config_files.append(str(json_file))

    # Sort by modification time (newest first)
    config_files.sort(key=os.path.getmtime, reverse=True)
    return config_files[:200]  # Limit to 200 most recent


def get_config_file_choices() -> List[str]:
    """Get list of config files for dropdown, formatted nicely"""
    files = scan_output_dirs_for_configs()
    choices = []
    for f in files:
        path = Path(f)
        # Format: "filename.json (parent_dir)"
        parent = path.parent.name
        choices.append(f"{path.name} ({parent})")
    return choices


def run_from_prompt_run(
    run_name: str,
    interleave_prompts: bool = True,
    start_at: int = 1,
    loop_back: bool = False,
    progress=gr.Progress()
):
    """
    Run batch rendering from a saved prompt run file.

    This skips the prompt generation step and uses pre-saved prompts.
    Seeds are generated fresh at render time (not stored in JSON).

    Args:
        run_name: Name of the saved prompt run file
        interleave_prompts: If True, alternate between different base themes/prompts
                           instead of rendering all variations of one theme first.
                           This lets you see variety early and identify good prompts faster.
        start_at: 1-based prompt number to start from (skip earlier prompts)
        loop_back: If True, loop back to start when finished (runs indefinitely until stopped)
    """
    global model, model_loaded, batch_running, batch_stop_requested, batch_results

    # Debug: Check model state
    print(f"[PROMPT_RUN CHECK] model_loaded={model_loaded}, model is None={model is None}")

    # Check if model exists (flag might not be set due to tokenizer error)
    if model is None:
        yield [], "Please load the model first!", "", []
        return

    # Fix flag if model exists but flag not set
    if not model_loaded and model is not None:
        print("[PROMPT_RUN] Model exists but flag was False - fixing flag")
        model_loaded = True

    if not run_name:
        yield [], "Please select a prompt run", "", []
        return

    if batch_running:
        yield [], "A batch is already running!", "", []
        return

    # Load the prompt run
    prompts, settings, status = load_prompt_run(run_name)
    if prompts is None:
        yield [], status, "", []
        return

    batch_running = True
    batch_stop_requested = False
    batch_results = []

    try:
        # Create batch output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_base_name = run_name.rsplit('_', 2)[0] if '_' in run_name else run_name.replace('.json', '')
        safe_name = "".join(c for c in run_base_name[:20] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        batch_dir = OUTPUT_DIR / "batches" / f"{safe_name}_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Get settings from the saved run
        images_per_combo = settings.get("images_per_combo", 1)
        aspect_ratio = settings.get("aspect_ratio", "1:1 (Square)")
        quality_preset = settings.get("quality", "Standard")
        negative_prompt = settings.get("negative_prompt", "")
        random_seeds = settings.get("random_seeds", True)  # Seeds generated at render time
        guidance_scale = settings.get("guidance_scale", 5.0)  # CFG guidance scale

        # Expand for images_per_combo and track original order
        all_jobs = []
        for prompt_idx, p in enumerate(prompts):
            for combo_idx in range(images_per_combo):
                all_jobs.append({
                    **p,
                    "combo_index": combo_idx,
                    "prompt_index": prompt_idx,  # Track original prompt position
                })

        # Interleave prompts if requested
        # Instead of: theme1_v1, theme1_v2, theme1_v3, theme2_v1, theme2_v2...
        # We get:     theme1_v1, theme2_v1, theme3_v1, theme1_v2, theme2_v2...
        if interleave_prompts and len(all_jobs) > 1:
            # Log order before sorting
            print(f"[INTERLEAVE] Before sort - first 6 jobs:")
            for i, j in enumerate(all_jobs[:6]):
                print(f"  [{i}] prompt_idx={j['prompt_index']}, combo_idx={j['combo_index']}, prompt='{j['prompt'][:40]}...'")

            # Sort by combo_index first (round), then by prompt_index
            # This groups all first variations together, then all second variations, etc.
            all_jobs.sort(key=lambda x: (x["combo_index"], x["prompt_index"]))

            # Log order after sorting
            print(f"\n[INTERLEAVE] After sort - first 12 jobs (should alternate prompts):")
            for i, j in enumerate(all_jobs[:12]):
                print(f"  [{i}] prompt_idx={j['prompt_index']}, combo_idx={j['combo_index']}, prompt='{j['prompt'][:40]}...'")
            print(f"[INTERLEAVE] Reordered {len(all_jobs)} jobs to alternate between prompts")
        elif not interleave_prompts:
            print(f"[INTERLEAVE] Disabled - jobs will run sequentially by prompt")

        total_jobs = len(all_jobs)

        # Get rendering settings
        image_size = ASPECT_RATIOS.get(aspect_ratio, "1024x1024")
        steps = QUALITY_PRESETS.get(quality_preset, {}).get("steps", 20)

        # Save manifest
        manifest = {
            "source_run": run_name,
            "interleaved": interleave_prompts,
            "created_at": datetime.now().isoformat(),
            "total_images": total_jobs,
            "settings": settings,
            "prompts": prompts
        }
        with open(batch_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        # Handle start_at (1-based to 0-based index)
        start_at = max(1, int(start_at)) if start_at else 1
        start_idx = min(start_at - 1, total_jobs - 1)  # Clamp to valid range

        # Info about start position and loop
        start_info = ""
        if start_idx > 0:
            start_info = f" (starting at #{start_at})"
        if loop_back:
            start_info += " [LOOP MODE - runs until stopped]"

        yield [], f"Loaded {len(prompts)} prompts from {run_name}\nRendering {total_jobs} images{start_info}...", f"Output: {batch_dir}", []

        generated_images = []
        image_counter = 0  # Total images generated across all loops
        loop_count = 0

        # Main rendering loop with loop_back support
        while True:
            # Determine which jobs to process this iteration
            if loop_count == 0:
                # First pass: start from start_idx
                job_indices = list(range(start_idx, total_jobs))
            else:
                # Subsequent passes (loop_back): start from beginning
                job_indices = list(range(total_jobs))

            for idx in job_indices:
                if batch_stop_requested:
                    loop_info = f" (loop #{loop_count + 1})" if loop_back and loop_count > 0 else ""
                    yield generated_images, f"Stopped at image {image_counter}{loop_info}", f"Output: {batch_dir}", generated_images[-12:] if generated_images else []
                    break

                job = all_jobs[idx]
                prompt = job["prompt"]
                style = job["style"]

                # Use variation_seed if available (ensures same seed for all shots of same scene)
                if "variation_seed" in job:
                    seed = job["variation_seed"]
                    print(f"[PROMPT_RUN] Using variation_seed={seed} for consistent subject across shots")
                elif random_seeds:
                    seed = random.randint(0, 2**32 - 1)
                else:
                    seed = ((image_counter + idx) * 12345) % (2**32)

                # Process wildcards if present
                original_prompt = prompt
                if wildcard_available and wildcard_manager and wildcard_manager.has_wildcards(prompt):
                    prompt = wildcard_manager.process_prompt(prompt)

                # Apply style
                styled_prompt = apply_style(prompt, style)

                # Add negative prompt
                full_prompt = styled_prompt
                if negative_prompt and negative_prompt.strip():
                    full_prompt = f"{styled_prompt} --no {negative_prompt.strip()}"

                # Progress display with loop info
                loop_info = f" (loop #{loop_count + 1})" if loop_back and loop_count > 0 else ""
                progress_pct = (image_counter + 1) / total_jobs if not loop_back else 0.5  # Indeterminate for loop mode
                progress(progress_pct, desc=f"Image {image_counter + 1}, prompt #{idx + 1}/{total_jobs}{loop_info}")
                yield generated_images, f"Generating image {image_counter + 1}, prompt #{idx + 1}/{total_jobs}{loop_info}: {prompt[:50]}...", f"Output: {batch_dir}", generated_images[-12:] if generated_images else []

                try:
                    start_time = time.time()

                    # Log diagnostics
                    log_render_diagnostics(
                        prompt=full_prompt,
                        seed=seed,
                        image_size=image_size,
                        steps=steps,
                        negative_prompt=negative_prompt,
                        style=style,
                        batch_idx=idx,
                        batch_total=total_jobs,
                        session_type="prompt_run",
                        guidance_scale=guidance_scale
                    )

                    print(f"[RENDER] Starting image generation #{image_counter + 1}...")
                    print(f"[RENDER] Prompt: {full_prompt[:100]}...")
                    print(f"[RENDER] Size: {image_size}, Steps: {steps}, Seed: {seed}")

                    # Check if tokenizer is still loaded (can get lost due to memory pressure)
                    if not hasattr(model, '_tkwrapper') or model._tkwrapper is None:
                        print("[RENDER] WARNING: Tokenizer missing, attempting to reload...")
                        model_id = "/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage3-SDNQ"
                        try:
                            model.load_tokenizer(model_id)
                            print("[RENDER] Tokenizer reloaded successfully!")
                        except Exception as tok_err:
                            print(f"[RENDER] Tokenizer reload failed: {tok_err}")
                            raise RuntimeError(f"Tokenizer not loaded and reload failed: {tok_err}")

                    image = model.generate_image(
                        prompt=full_prompt,
                        seed=seed,
                        image_size=image_size,
                        diff_infer_steps=steps,
                        diff_guidance_scale=guidance_scale,
                        stream=True,
                    )

                    print(f"[RENDER] Image generated successfully!")
                    gen_time = time.time() - start_time

                    # Save image with unique counter to avoid overwrites in loop mode
                    safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
                    filename = f"{image_counter + 1:04d}_{safe_prompt}_s{seed}.png"
                    filepath = batch_dir / filename
                    image.save(filepath)

                    # Save config
                    wildcards_used = (original_prompt != prompt)
                    config = {
                        "prompt": original_prompt,
                        "processed_prompt": prompt if wildcards_used else None,
                        "styled_prompt": styled_prompt,
                        "negative_prompt": negative_prompt if negative_prompt else None,
                        "style": style,
                        "seed": seed,
                        "image_size": image_size,
                        "aspect_ratio": aspect_ratio,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "quality_preset": quality_preset,
                        "source_run": run_name,
                        "batch_index": image_counter + 1,
                        "prompt_index": idx + 1,
                        "loop_number": loop_count + 1 if loop_back else None,
                        "wildcards_used": wildcards_used,
                        "generation_time": gen_time
                    }
                    save_image_config(str(filepath), config)

                    generated_images.append(str(filepath))
                    batch_results.append({
                        "index": image_counter + 1,
                        "prompt_index": idx + 1,
                        "prompt": prompt,
                        "style": style,
                        "seed": seed,
                        "filepath": str(filepath),
                        "generation_time": gen_time,
                        "loop": loop_count + 1 if loop_back else None
                    })

                    image_counter += 1

                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"[RENDER ERROR] Image #{image_counter + 1} failed: {str(e)}")
                    print(f"[RENDER TRACEBACK]\n{error_details}")
                    batch_results.append({
                        "index": image_counter + 1,
                        "prompt_index": idx + 1,
                        "prompt": prompt,
                        "style": style,
                        "error": str(e)
                    })
                    image_counter += 1

                    # Clean up GPU memory after error (helps with OOM recovery)
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"[MEMORY] Cleared cache after error")

            # End of inner for loop - check if we should continue
            if batch_stop_requested:
                break

            # If not looping, we're done after first pass
            if not loop_back:
                break

            # Loop back: increment loop count and continue
            loop_count += 1
            print(f"[LOOP] Completed loop #{loop_count}, starting loop #{loop_count + 1}")

        # Save results
        with open(batch_dir / "results.json", 'w') as f:
            json.dump(batch_results, f, indent=2)

        completed = len([r for r in batch_results if "filepath" in r])
        failed = len([r for r in batch_results if "error" in r])

        loop_info = f" ({loop_count + 1} loops)" if loop_back and loop_count > 0 else ""
        final_status = f"Complete! {completed} images from {run_name}{loop_info}"
        if failed > 0:
            final_status += f", {failed} failed"
        final_status += f"\nOutput: {batch_dir}"

        yield generated_images, final_status, str(batch_dir), generated_images[-12:] if generated_images else []

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[PROMPT_RUN ERROR] {str(e)}")
        print(f"[PROMPT_RUN TRACEBACK]\n{error_details}")
        yield [], f"Error: {str(e)}\n\n{error_details}", "", []

    finally:
        batch_running = False


def generate_prompts_only(
    batch_name: str,
    themes_text: str,
    variations_per_theme: int,
    styles_selected: List[str],
    images_per_combo: int,
    llm_backend: str,
    ollama_model: str,
    enhance_prompts: bool,
    aspect_ratio: str,
    quality_preset: str,
    random_seeds: bool,
    negative_prompt: str,
    ollama_length: str,
    ollama_complexity: str,
    guidance_scale: float = 5.0,
    progress=gr.Progress()
):
    """
    Generate prompts using LLM and save them WITHOUT rendering.

    This allows generating many prompt batches while using GPU for LLM,
    then later loading the image model to render from saved prompts.
    """
    if not themes_text.strip():
        yield "Please enter at least one theme"
        return

    if not styles_selected:
        yield "Please select at least one style"
        return

    progress(0.1, desc=f"Generating prompts with {llm_backend}...")
    yield f"Generating prompt variations with {llm_backend}..."

    try:
        # Generate all prompt combinations (no rendering)
        prompts = generate_batch_prompts(
            themes_text, variations_per_theme, styles_selected,
            llm_backend, ollama_model, enhance_prompts,
            ollama_length, ollama_complexity
        )

        if not prompts:
            yield "No prompts generated. Check your themes and LLM settings."
            return

        progress(0.8, desc="Saving prompts...")

        # Save the prompt run
        settings = {
            "themes": themes_text.split('\n'),
            "variations_per_theme": variations_per_theme,
            "styles": styles_selected,
            "images_per_combo": images_per_combo,
            "aspect_ratio": aspect_ratio,
            "quality": quality_preset,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "ollama_model": ollama_model,
            "ollama_length": ollama_length,
            "ollama_complexity": ollama_complexity,
            "enhance_prompts": enhance_prompts,
            "random_seeds": random_seeds
        }

        prompt_run_path = save_prompt_run(batch_name, prompts, settings)
        filename = Path(prompt_run_path).name

        # Calculate total images that would be generated
        total_images = len(prompts) * images_per_combo

        progress(1.0, desc="Done!")

        yield f"""**Prompts Generated & Saved!**

**File:** `{filename}`
**Location:** `outputs/prompt_runs/`

**Summary:**
- {len(prompts)} unique prompts
- {len(styles_selected)} styles
- {images_per_combo} images per combo
- **{total_images} total images** when rendered

**Next steps:**
1. Edit the JSON file if needed
2. Load the image model (click "Load Image Model" above)
3. Go to "Saved Prompt Runs" tab
4. Select this run and click "Run Selected Prompts"
"""

    except Exception as e:
        yield f"Error generating prompts: {str(e)}"


# ============================================================================
# BATCH GALLERY BROWSER
# ============================================================================

# Store custom batch directory path globally
custom_batch_base_dir = None


def get_batch_directories(custom_dir: str = None) -> List[str]:
    """Get list of batch output directories"""
    global custom_batch_base_dir

    if custom_dir:
        custom_batch_base_dir = custom_dir

    batch_dir = Path(custom_batch_base_dir) if custom_batch_base_dir else OUTPUT_DIR / "batches"

    if not batch_dir.exists():
        return ["(No batches yet)"]

    # Check if this directory itself is a batch (contains .png files)
    if list(batch_dir.glob("*.png")):
        # This directory IS a batch, return its name
        return [batch_dir.name]

    # Otherwise, look for subdirectories that are batches
    dirs = sorted(
        [d.name for d in batch_dir.iterdir() if d.is_dir()],
        reverse=True
    )
    return dirs if dirs else ["(No batches yet)"]


def load_custom_batch_directory(custom_path: str):
    """Load batches from a custom directory path"""
    global custom_batch_base_dir

    if not custom_path or not custom_path.strip():
        # Reset to default
        custom_batch_base_dir = None
        dirs = get_batch_directories()
        return (
            gr.update(choices=dirs, value=dirs[0] if dirs else None),
            f"Reset to default: {OUTPUT_DIR / 'batches'}"
        )

    path = Path(custom_path.strip())

    if not path.exists():
        return gr.update(), f"Error: Directory does not exist: {custom_path}"

    if not path.is_dir():
        return gr.update(), f"Error: Path is not a directory: {custom_path}"

    # Check if this is a batch directory (has PNG images)
    has_images = list(path.glob("*.png"))

    if has_images:
        # This directory itself is a batch
        custom_batch_base_dir = str(path.parent)
        dirs = [path.name]
        return (
            gr.update(choices=dirs, value=path.name),
            f"Loaded batch directory: {path.name} ({len(has_images)} images)"
        )

    # Check for subdirectories with images
    subdirs = [d for d in path.iterdir() if d.is_dir() and list(d.glob("*.png"))]

    if subdirs:
        custom_batch_base_dir = str(path)
        dirs = sorted([d.name for d in subdirs], reverse=True)
        return (
            gr.update(choices=dirs, value=dirs[0] if dirs else None),
            f"Found {len(dirs)} batch folders in: {path}"
        )

    return gr.update(), f"No batch folders found in: {path}"


def load_batch_from_file(file_path: str):
    """Load batch directory from a picked file - extracts parent directory"""
    if not file_path:
        return gr.update(), gr.update(), "No file selected"

    path = Path(file_path)

    if not path.exists():
        return gr.update(), gr.update(), f"File not found: {file_path}"

    # Get the parent directory of the file
    batch_dir = path.parent

    # Load this directory as a batch
    result = load_custom_batch_directory(str(batch_dir))

    # Also update the textbox with the path
    return result[0], gr.update(value=str(batch_dir)), result[1]


def get_batch_images(batch_name: str, page: int = 0, per_page: int = 24) -> tuple:
    """Get images from a specific batch directory with pagination"""
    global custom_batch_base_dir

    if not batch_name or batch_name == "(No batches yet)":
        return [], "No batch selected", 0, 0

    # Use custom base dir if set, otherwise use default
    base_dir = Path(custom_batch_base_dir) if custom_batch_base_dir else OUTPUT_DIR / "batches"
    batch_path = base_dir / batch_name

    if not batch_path.exists():
        return [], f"Batch not found: {batch_name}", 0, 0

    all_images = sorted(batch_path.glob("*.png"))
    total_images = len(all_images)
    total_pages = max(1, (total_images + per_page - 1) // per_page)

    page = max(0, min(page, total_pages - 1))
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, total_images)

    page_images = [str(img) for img in all_images[start_idx:end_idx]]

    location = f"({base_dir})" if custom_batch_base_dir else "(default)"
    info = f"Batch: {batch_name} {location}\nImages: {start_idx + 1}-{end_idx} of {total_images} | Page {page + 1}/{total_pages}"

    return page_images, info, page, total_pages


def load_image_from_gallery(evt: gr.SelectData, current_batch: str):
    """Load config from a selected gallery image into the main UI"""
    global custom_batch_base_dir

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
        # Use custom base dir if set, otherwise use default
        base_dir = Path(custom_batch_base_dir) if custom_batch_base_dir else OUTPUT_DIR / "batches"
        batch_dir = base_dir / current_batch
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
        return [gr.update()] * 9 + ["No file selected"]

    try:
        config = load_image_config(config_file.name if hasattr(config_file, 'name') else config_file)

        if "error" in config:
            return [gr.update()] * 9 + [f"Error: {config['error']}"]

        # Return updates for: prompt, style, aspect_ratio, quality, steps, seed, use_random, use_ollama, negative_prompt, status
        return [
            gr.update(value=config.get('prompt', '')),
            gr.update(value=config.get('style', 'None')),
            gr.update(value=config.get('aspect_ratio', '1:1 (Square)')),
            gr.update(value=config.get('quality_preset', 'Standard')),
            gr.update(value=config.get('steps', 20)),
            gr.update(value=config.get('seed', 0)),
            gr.update(value=False),  # Uncheck random seed to use the saved seed
            gr.update(value=config.get('use_ollama', False)),
            gr.update(value=config.get('negative_prompt', '')),
            f"Loaded config from: {config.get('image_path', 'unknown')}\nSeed: {config.get('seed')}"
        ]
    except Exception as e:
        return [gr.update()] * 9 + [f"Error loading config: {e}"]


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
    guidance_scale: float = 5.0,
    progress=gr.Progress()
):
    """Generate an image from the text prompt. Supports batch generation."""
    global model, model_loaded

    # Debug: Check model state
    print(f"[GENERATE CHECK] model_loaded={model_loaded}, model is None={model is None}")

    # Check if model exists (flag might not be set due to tokenizer error)
    if model is None:
        yield None, "Please load the model first! Click 'Load Model' above.", "", None
        return

    # Fix flag if model exists but flag not set
    if not model_loaded and model is not None:
        print("[GENERATE] Model exists but flag was False - fixing flag")
        model_loaded = True

    if not prompt.strip():
        yield None, "Please enter a prompt!", "", None
        return

    batch_count = int(batch_count)
    original_prompt = prompt.strip()

    # Reset stop flag at start
    global single_generation_stop
    single_generation_stop = False

    try:
        # Determine image size (same for all batch items)
        if custom_size and custom_size.strip():
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
            # Check for stop request
            if single_generation_stop:
                completed = batch_idx
                yield last_image, f"Stopped at {completed}/{batch_count} images", "\n".join(all_info), last_seed
                reset_single_generation_stop()
                return

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
                    # Don't pass seed - let wildcards be truly random for each batch item
                    processed_prompt = wildcard_manager.process_prompt(original_prompt)

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
            if negative_prompt and negative_prompt.strip():
                full_prompt = f"{styled_prompt}. Avoid: {negative_prompt.strip()}"

            progress((batch_idx * 0.9 / batch_count) + 0.05, desc=f"{batch_label}Initializing...")
            yield last_image, f"{batch_label}Starting generation...", "", current_seed

            start_time = time.time()

            # Log comprehensive diagnostics before each render
            log_render_diagnostics(
                prompt=full_prompt,
                seed=current_seed,
                image_size=image_size,
                steps=inference_steps,
                negative_prompt=negative_prompt,
                style=style,
                batch_idx=batch_idx,
                batch_total=batch_count,
                use_ollama=use_ollama,
                ollama_model=ollama_model,
                session_type="single" if batch_count == 1 else "batch_count",
                guidance_scale=guidance_scale
            )

            progress((batch_idx * 0.9 / batch_count) + 0.1, desc=f"{batch_label}Generating with {inference_steps} steps...")
            yield last_image, f"{batch_label}Generating image ({inference_steps} steps)...", "", current_seed

            # Generate the image
            image = model.generate_image(
                prompt=full_prompt,
                seed=current_seed,
                image_size=image_size,
                diff_infer_steps=inference_steps,
                diff_guidance_scale=guidance_scale,
                stream=True,
            )

            generation_time = time.time() - start_time

            # Generate filename and save if requested
            # Use session directory like batch generation does
            session_dir = get_or_create_session_dir(original_prompt)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip()
            safe_prompt = safe_prompt.replace(" ", "_")
            # Use batch-like numbering within the session
            filename = f"{batch_idx + 1:04d}_{safe_prompt}_s{current_seed}.png"
            filepath = session_dir / filename

            if save_image:
                image.save(filepath)
                print(f"[SAVED] {filepath}")

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
                    "guidance_scale": guidance_scale,
                    "quality_preset": quality_preset,
                    "use_ollama": use_ollama,
                    "ollama_model": ollama_model if use_ollama else None,
                    "wildcards_used": wildcards_used,
                    "generation_time": generation_time,
                    "batch_index": batch_idx + 1 if batch_count > 1 else None,
                    "batch_total": batch_count if batch_count > 1 else None,
                    "session_dir": str(session_dir)
                }
                config_path = save_image_config(str(filepath), config)

                save_status = f"Saved: {session_dir.name}/{filename}"
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
        session_info = f"Output: {session_dir}" if save_image else ""
        if batch_count > 1:
            total_info = f"Batch complete: {batch_count} images generated\n{session_info}\n\n" + "\n".join(all_info)
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
{save_status if save_image else "Not saved (enable 'Auto-save' to save)"}
{session_info}"""
            else:
                total_info = f"""Prompt: {prompt}
Style: {style}
Size: {image_size}
Steps: {inference_steps}
Seed: {last_seed}
Time: {generation_time:.1f}s
{save_status if save_image else "Not saved (enable 'Auto-save' to save)"}
{session_info}"""
            final_status = "Generation complete!"

        # Reset session for next generation batch
        reset_session_dir()

        yield last_image, final_status, total_info, last_seed

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        yield None, error_msg, "", seed


def reroll_with_new_seed(prompt, negative_prompt, style, aspect_ratio,
                         custom_size, quality_preset, custom_steps,
                         use_custom_steps, save_image, use_ollama,
                         ollama_model, ollama_length, ollama_complexity,
                         guidance_scale):
    """Regenerate with a completely new random seed (same prompt settings)."""
    new_seed = random.randint(0, 2**32 - 1)

    yield from generate_image(
        prompt, negative_prompt, style, aspect_ratio, custom_size,
        quality_preset, custom_steps, use_custom_steps,
        new_seed, False, save_image, use_ollama, ollama_model,
        1, ollama_length, ollama_complexity, guidance_scale
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


# Instructions content for the help accordion
INSTRUCTIONS_MARKDOWN = """
### Wildcard System

Use `[category]` syntax in your prompts for random substitution:

**Basic:** `A [animal] in a [landscape]` → "A tiger in a forest"

**Combined:** `A [color+animal]` → "A golden dragon"

Categories use prefixes like `artist-famous`, `color-warm`, `mood-dark`.
Click **Insert Wildcard** above to browse all available categories.

---

### Prompting Tips for HunyuanImage 3.0

This model is different from Stable Diffusion and FLUX:

- **Write prose, not keywords** — Full sentences work better than comma lists
- **No bracket weights** — `(word:1.4)` syntax does NOT work here
- **Embed negatives** — Use "no watermark, no text" in your prompt
- **Use quotes for text** — `with the text "HELLO" in bold serif`
- **Be specific** — "85mm lens at f/2.8, golden hour" beats "professional photo"

---

### Prompt Structure

`[Subject] + [Details] + [Setting] + [Lighting] + [Style]`

**Good:** "A weathered lighthouse keeper in his 60s with a salt-and-pepper beard,
standing in a Victorian lighthouse. Golden hour light streaming through windows,
creating dramatic rim lighting. Photorealistic, 35mm film grain."

**Bad:** "lighthouse keeper, old man, beard, beautiful, amazing, 8k, masterpiece"

---

### Ollama Enhancement

Enable **"Enhance with Ollama"** to expand simple prompts into detailed descriptions.
Larger models (24B+) produce more creative results but are slower.

---

### Settings Reference

| CFG Scale | Effect |
|-----------|--------|
| 1-3 | Creative/abstract |
| **5 (default)** | Balanced |
| 8-15 | Strict adherence |

| Quality | Steps | Use Case |
|---------|-------|----------|
| Fast | 15 | Quick tests |
| **Standard** | 20 | Normal use |
| High | 30 | Final renders |
"""


# Build the Gradio interface
def create_ui():
    """Create and configure the Gradio interface."""

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
        css=CUSTOM_CSS,
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

                # Model status section - prominent load/unload controls
                with gr.Group():
                    gr.Markdown("### Image Model Control")
                    model_status = gr.Markdown(
                        value=get_model_status(),
                    )
                    with gr.Row():
                        load_model_btn = gr.Button(
                            "Load Image Model",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                        unload_model_btn = gr.Button(
                            "Unload Model",
                            variant="stop",
                            size="lg",
                            scale=1
                        )
                        refresh_status_btn = gr.Button(
                            "↻",
                            size="lg",
                            scale=0
                        )
                    gr.Markdown("*Unload model to free GPU for large LLM prompt generation*", elem_classes=["info-text"])

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

                # Style Editor Accordion
                with gr.Accordion("Edit Style Presets", open=False):
                    gr.Markdown("**Add, edit, or delete style presets.** Changes are saved automatically.")

                    with gr.Row():
                        style_editor_dropdown = gr.Dropdown(
                            label="Select Style to Edit",
                            choices=list(STYLE_PRESETS.keys()),
                            value="None",
                            scale=2
                        )
                        style_refresh_btn = gr.Button("↻", size="sm", scale=0)

                    style_name_input = gr.Textbox(
                        label="Style Name",
                        placeholder="My Custom Style",
                        lines=1
                    )

                    style_suffix_input = gr.Textbox(
                        label="Style Suffix (added to prompt)",
                        placeholder=", cinematic lighting, dramatic atmosphere, 8k quality",
                        lines=2,
                        info="This text is appended to your prompt when this style is selected"
                    )

                    with gr.Row():
                        style_add_btn = gr.Button("Add/Update Style", variant="primary", size="sm")
                        style_delete_btn = gr.Button("Delete Style", variant="stop", size="sm")
                        style_reset_btn = gr.Button("Reset to Defaults", variant="secondary", size="sm")

                    style_editor_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1
                    )

                    with gr.Accordion("Bulk Edit (Import/Export)", open=False):
                        gr.Markdown("*Edit all styles as text. Format: `StyleName: suffix text`*")
                        style_bulk_text = gr.Textbox(
                            label="All Styles (one per line)",
                            lines=10,
                            placeholder="Photorealistic: , photorealistic, hyperrealistic, 8k\nCinematic: , cinematic lighting, dramatic"
                        )
                        with gr.Row():
                            style_export_btn = gr.Button("Export Current Styles", size="sm")
                            style_import_btn = gr.Button("Import from Text", variant="primary", size="sm")

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

                # Instructions accordion
                with gr.Accordion("Instructions & Tips", open=False):
                    gr.Markdown(INSTRUCTIONS_MARKDOWN)

                # Main Generation Settings (Always Visible)
                with gr.Group():
                    gr.Markdown("### Generation Settings")
                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="1:1 (Square)",
                            scale=2
                        )
                        quality_preset = gr.Radio(
                            label="Quality",
                            choices=list(QUALITY_PRESETS.keys()),
                            value="Standard",
                            scale=3
                        )
                    with gr.Row():
                        use_random_seed = gr.Checkbox(
                            label="Random Seed",
                            value=True,
                            scale=1
                        )
                        seed = gr.Number(
                            label="Seed",
                            value=0,
                            precision=0,
                            scale=2,
                            interactive=True
                        )

                # Advanced settings in tabs
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Tabs():
                        with gr.Tab("Custom Size"):
                            custom_size = gr.Textbox(
                                label="Custom Size (optional)",
                                placeholder="e.g., 1024x768",
                                info="Overrides aspect ratio if set"
                            )
                            gr.Markdown("*Leave blank to use the Aspect Ratio above*")

                        with gr.Tab("Custom Steps"):
                            gr.Markdown("*Override the Quality preset with custom steps:*")
                            with gr.Row():
                                use_custom_steps = gr.Checkbox(
                                    label="Use custom steps",
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
                            gr.Markdown("---")
                            gr.Markdown("**Guidance Scale (CFG)** - How closely to follow the prompt")
                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=1.0,
                                    maximum=15.0,
                                    value=5.0,
                                    step=0.5,
                                    info="Higher = follows prompt more strictly (default: 5.0)"
                                )
                            gr.Markdown("*Low (1-3): More creative/abstract | Medium (4-7): Balanced | High (8-15): Strict prompt adherence*")

                        with gr.Tab("Seed Info"):
                            gr.Markdown("*Same seed + same prompt = same image*")
                            gr.Markdown("Use the Seed field above to reproduce specific images.")

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

                # Save/Load Prompt Configs (like batch)
                with gr.Accordion("Save/Load Prompt Configs", open=False):
                    gr.Markdown("**Save your current settings to reuse later**")
                    with gr.Row():
                        prompt_config_name = gr.Textbox(
                            label="Config Name",
                            placeholder="my_portrait_settings",
                            scale=3
                        )
                    with gr.Row():
                        save_prompt_config_btn = gr.Button("Save", variant="primary", size="sm", scale=1)
                        overwrite_prompt_config_btn = gr.Button("Overwrite", variant="secondary", size="sm", scale=1)
                    with gr.Row():
                        prompt_config_dropdown = gr.Dropdown(
                            label="Saved Configs",
                            choices=get_saved_prompt_configs(),
                            interactive=True,
                            scale=3
                        )
                        refresh_prompt_configs_btn = gr.Button("↻", size="sm", scale=0)
                    with gr.Row():
                        load_prompt_config_btn = gr.Button("Load", variant="primary", size="sm", scale=1)
                        delete_prompt_config_btn = gr.Button("Delete", variant="stop", size="sm", scale=1)
                    prompt_config_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2
                    )
                    gr.Markdown("*Saves: prompt, style, aspect ratio, quality, seed settings, Ollama settings, batch count*")

                # Legacy: Load from image JSON sidecar
                with gr.Accordion("Load from Image Config File", open=False):
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
                        maximum=1000,
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
                    stop_generation_btn = gr.Button(
                        "Stop",
                        variant="stop",
                        size="lg",
                        scale=1
                    )
                with gr.Row():
                    reroll_btn = gr.Button(
                        "Re-roll Seed",
                        size="sm",
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
                                lines=6,
                                max_lines=20
                            )

                            batch_negative_prompt = gr.Textbox(
                                label="Negative Prompt (applied to all images)",
                                placeholder="blurry, low quality, distorted, ugly, bad anatomy, watermark, text",
                                lines=2,
                                info="What to avoid in all generated images"
                            )

                            with gr.Row():
                                batch_variations = gr.Slider(
                                    label="Variations per theme",
                                    minimum=1,
                                    maximum=500,
                                    value=3,
                                    step=1,
                                    info="Ollama generates this many variations of each theme"
                                )
                                batch_images_per = gr.Slider(
                                    label="Images per combo",
                                    minimum=1,
                                    maximum=500,
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
                                batch_guidance_scale = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=1.0,
                                    maximum=15.0,
                                    value=5.0,
                                    step=0.5,
                                    info="CFG: Higher = follows prompt more strictly"
                                )

                            with gr.Row():
                                batch_llm_backend = gr.Dropdown(
                                    label="LLM Backend",
                                    choices=["Ollama", "LM Studio"],
                                    value="Ollama",
                                    info="Which LLM server to use for prompt enhancement"
                                )
                                batch_enhance = gr.Checkbox(
                                    label="Enhance prompts",
                                    value=True,
                                    info="Use LLM to enhance each prompt"
                                )

                            with gr.Row():
                                batch_ollama_model = gr.Dropdown(
                                    label="Ollama Model",
                                    choices=get_ollama_models_list() if ollama_available else OLLAMA_MODELS,
                                    value="qwen2.5:7b-instruct",
                                    visible=True
                                )
                                batch_lmstudio_status = gr.Textbox(
                                    label="LM Studio Model",
                                    value="(Select LM Studio backend to check)",
                                    interactive=False,
                                    visible=False
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
                                generate_prompts_only_btn = gr.Button(
                                    "Generate Prompts Only",
                                    variant="secondary",
                                    size="lg",
                                    elem_id="gen-prompts-btn"
                                )
                            with gr.Row():
                                batch_start_btn = gr.Button("Start Batch", variant="primary", size="lg")
                                batch_stop_btn = gr.Button("Stop", variant="stop", size="sm")
                            gr.Markdown("*'Generate Prompts Only' saves to prompt_runs/ without rendering*")

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

                            # Enhanced Save/Load batch configs
                            gr.Markdown("---\n**Configuration Management**")

                            with gr.Row():
                                config_name_input = gr.Textbox(
                                    label="Config Name",
                                    placeholder="Enter name to save as...",
                                    scale=2
                                )
                                batch_save_btn = gr.Button("Save", variant="primary", size="sm", scale=1)
                                batch_save_overwrite_btn = gr.Button("Overwrite", variant="secondary", size="sm", scale=1)

                            with gr.Row():
                                batch_config_dropdown = gr.Dropdown(
                                    label="Saved Configs",
                                    choices=bm.get_config_choices() if batch_manager_available else get_saved_batch_configs(),
                                    value=None,
                                    allow_custom_value=False,
                                    scale=3
                                )
                                batch_load_btn = gr.Button("Load", variant="primary", size="sm", scale=1)
                                batch_refresh_configs_btn = gr.Button("↻", size="sm", scale=0)

                            with gr.Row():
                                batch_delete_config_btn = gr.Button("Delete Config", variant="stop", size="sm")
                                batch_rename_input = gr.Textbox(
                                    label="New Name",
                                    placeholder="New name...",
                                    scale=2,
                                    visible=True
                                )
                                batch_rename_btn = gr.Button("Rename", size="sm")

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

                # Tab 2: Load & Run Saved Prompt Runs
                with gr.Tab("Saved Prompt Runs"):
                    gr.Markdown("""
                    **Load and run previously generated prompt lists**

                    After batch prompt generation, prompts are saved to `outputs/prompt_runs/`.
                    You can edit these JSON files and re-run them as many times as needed.
                    **Seeds are generated fresh each run** - run the same prompts multiple times for variety!
                    """)

                    with gr.Row():
                        # Left side - Run selection and controls
                        with gr.Column(scale=1):
                            with gr.Row():
                                prompt_run_dropdown = gr.Dropdown(
                                    label="Select Prompt Run",
                                    choices=get_saved_prompt_runs(),
                                    interactive=True,
                                    scale=3
                                )
                                refresh_prompt_runs_btn = gr.Button("↻", size="sm", scale=0)

                            # Rendering options
                            with gr.Row():
                                interleave_checkbox = gr.Checkbox(
                                    label="Interleave prompts",
                                    value=True,
                                    info="Alternate between themes (1,2,3,1,2,3...) instead of sequential (1,1,1,2,2,2...)"
                                )

                            with gr.Row():
                                start_at_prompt = gr.Number(
                                    label="Start at prompt #",
                                    value=1,
                                    minimum=1,
                                    step=1,
                                    info="Skip to this prompt number (1-based)"
                                )
                                loop_back_checkbox = gr.Checkbox(
                                    label="Loop back",
                                    value=False,
                                    info="When finished, loop back to start and continue indefinitely"
                                )

                            with gr.Row():
                                run_from_saved_btn = gr.Button(
                                    "Run Selected Prompts",
                                    variant="primary",
                                    size="lg"
                                )
                                stop_prompt_run_btn = gr.Button(
                                    "Stop",
                                    variant="stop",
                                    size="lg"
                                )

                            with gr.Row():
                                delete_prompt_run_btn = gr.Button(
                                    "Delete Run",
                                    variant="stop",
                                    size="sm"
                                )

                            prompt_run_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                lines=3
                            )

                            prompt_run_output_dir = gr.Textbox(
                                label="Output Directory",
                                interactive=False
                            )

                            gr.Markdown("""
                            **How it works:**
                            - **Seeds**: Generated fresh each run (not saved in JSON)
                            - **Interleave**: See all themes early, find best prompts faster
                            - **Start at #**: Skip to a specific prompt (resume mid-batch)
                            - **Loop back**: Run indefinitely until stopped, cycling through prompts
                            - **Edit JSON**: Modify `outputs/prompt_runs/*.json` before running
                            """)

                        # Right side - Preview
                        with gr.Column(scale=1):
                            prompt_run_preview = gr.Markdown(
                                value="Select a prompt run to preview...",
                                label="Preview"
                            )

                    # Gallery for prompt run results
                    gr.Markdown("---\n**Rendering Progress**")
                    prompt_run_gallery = gr.Gallery(
                        label="Generated Images",
                        columns=6,
                        rows=2,
                        height=200,
                        object_fit="cover"
                    )

                # Tab 3: Combine Configs into Batch
                with gr.Tab("Combine Configs"):
                    gr.Markdown("""
                    **Combine JSON configs from individual images into a batch**

                    Select multiple `.json` config files from your outputs folder (sidecar files from individual generations)
                    and combine them into a prompt run that can be re-rendered with new seeds.
                    """)

                    with gr.Row():
                        # Left side - File selection and options
                        with gr.Column(scale=1):
                            combine_file_upload = gr.File(
                                label="Upload JSON Config Files",
                                file_count="multiple",
                                file_types=[".json"],
                                type="filepath"
                            )

                            gr.Markdown("**Options:**")
                            with gr.Row():
                                combine_run_name = gr.Textbox(
                                    label="Batch Name",
                                    value="combined_batch",
                                    placeholder="Name for the new prompt run"
                                )
                            with gr.Row():
                                combine_images_per = gr.Slider(
                                    label="Images per Prompt",
                                    minimum=1,
                                    maximum=50,
                                    value=1,
                                    step=1,
                                    info="How many images to generate for each config"
                                )
                            with gr.Row():
                                combine_use_seeds = gr.Checkbox(
                                    label="Store original seeds",
                                    value=False,
                                    info="Save original seeds for reference (new seeds still generated at render time)"
                                )

                            with gr.Row():
                                combine_create_btn = gr.Button(
                                    "Create Prompt Run",
                                    variant="primary",
                                    size="lg"
                                )
                                combine_clear_btn = gr.Button(
                                    "Clear",
                                    variant="secondary",
                                    size="sm"
                                )

                            combine_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                lines=3
                            )

                        # Right side - Preview
                        with gr.Column(scale=1):
                            combine_preview = gr.Markdown(
                                value="Upload JSON config files to preview...",
                                label="Preview"
                            )

                    gr.Markdown("""
                    ---
                    **How it works:**
                    - Upload the `.json` sidecar files generated alongside your images
                    - These contain the prompt, style, and settings used for each image
                    - The combined prompt run appears in the "Saved Prompt Runs" tab
                    - Run it to generate new variations with fresh seeds
                    """)

                # Tab 4: Browse Batches (Gallery) - Enhanced with Image Management
                with gr.Tab("Browse Batches"):
                    gr.Markdown("**Browse batches - Click images to view/edit, use batch operations below**")

                    # Batch selection row
                    with gr.Row():
                        browse_batch_dropdown = gr.Dropdown(
                            label="Select Batch",
                            choices=get_batch_directories(),
                            value=get_batch_directories()[0] if get_batch_directories() else None,
                            scale=3
                        )
                        browse_refresh_btn = gr.Button("↻ Refresh", size="sm", scale=1)

                    # Batch-level operations
                    with gr.Row():
                        browse_load_settings_btn = gr.Button("Load Batch Settings", variant="secondary", size="sm")
                        browse_delete_batch_btn = gr.Button("Delete Entire Batch", variant="stop", size="sm")

                    with gr.Accordion("Browse External Directory", open=False):
                        gr.Markdown("*Load batches from a different location:*")
                        with gr.Row():
                            browse_file_picker = gr.File(
                                label="Pick any image from the batch folder",
                                file_types=["image"],
                                type="filepath",
                                scale=3
                            )
                            browse_from_file_btn = gr.Button("Load from File", variant="secondary", size="sm", scale=1)
                        gr.Markdown("*Or enter path manually:*")
                        with gr.Row():
                            custom_batch_dir = gr.Textbox(
                                label="Directory Path",
                                placeholder="/path/to/your/batch/folder",
                                scale=4
                            )
                            browse_custom_btn = gr.Button("Load Path", variant="primary", size="sm", scale=1)
                        custom_browse_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=1,
                            visible=True
                        )

                    browse_info = gr.Markdown("Select a batch to view images...")

                    # Main gallery
                    browse_gallery = gr.Gallery(
                        label="Click an image to view/edit its config",
                        columns=6,
                        rows=4,
                        height=400,
                        object_fit="cover",
                        allow_preview=True
                    )

                    # Pagination
                    with gr.Row():
                        browse_page_state = gr.State(value=0)
                        browse_total_pages = gr.State(value=1)
                        browse_prev_btn = gr.Button("← Previous", size="sm")
                        browse_page_info = gr.Markdown("Page 1 of 1")
                        browse_next_btn = gr.Button("Next →", size="sm")

                    # Selected image state
                    selected_image_path = gr.State(value=None)

                    # Image Management Section
                    with gr.Accordion("Selected Image Management", open=True):
                        with gr.Row():
                            # Left: Image info display
                            with gr.Column(scale=1):
                                selected_image_info = gr.Markdown(
                                    "Click an image above to select it...",
                                    label="Image Info"
                                )
                                with gr.Row():
                                    browse_use_settings_btn = gr.Button("Use in Generator", variant="primary", size="sm")
                                    browse_delete_image_btn = gr.Button("Delete Image", variant="stop", size="sm")

                            # Right: Config editor
                            with gr.Column(scale=1):
                                selected_image_config = gr.Code(
                                    label="Image Config (JSON - editable)",
                                    language="json",
                                    lines=10,
                                    interactive=True
                                )
                                with gr.Row():
                                    browse_save_config_btn = gr.Button("Save Config Changes", variant="primary", size="sm")
                                    browse_reset_config_btn = gr.Button("Reset", size="sm")

                    browse_load_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=2,
                        placeholder="Click an image to view and edit its configuration..."
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

                # Tab 4: Wildcard Editor
                with gr.Tab("Wildcard Editor"):
                    if wildcard_editor_available:
                        gr.Markdown("**Manage wildcards: Add, edit, delete categories and items. Use LLM to generate new wildcards.**")

                        with gr.Row():
                            # Left: Category management
                            with gr.Column(scale=1):
                                gr.Markdown("### Categories")

                                wc_category_dropdown = gr.Dropdown(
                                    label="Select Category",
                                    choices=we.get_category_list(),
                                    value=None,
                                    allow_custom_value=False
                                )

                                with gr.Row():
                                    wc_refresh_btn = gr.Button("↻", size="sm", scale=0)
                                    wc_stats_btn = gr.Button("Stats", size="sm")

                                wc_category_info = gr.Markdown("Select a category to view items...")

                                with gr.Accordion("Add New Category", open=False):
                                    wc_new_category_name = gr.Textbox(
                                        label="Category Name",
                                        placeholder="e.g., material-metal, color-warm"
                                    )
                                    wc_new_category_items = gr.Textbox(
                                        label="Initial Items (one per line)",
                                        placeholder="gold\nsilver\nbronze\ncopper",
                                        lines=4
                                    )
                                    wc_add_category_btn = gr.Button("Create Category", variant="primary")

                                with gr.Accordion("Category Actions", open=False):
                                    wc_rename_input = gr.Textbox(
                                        label="New Name",
                                        placeholder="Enter new category name"
                                    )
                                    with gr.Row():
                                        wc_rename_btn = gr.Button("Rename", size="sm")
                                        wc_delete_category_btn = gr.Button("Delete Category", variant="stop", size="sm")

                            # Right: Items management
                            with gr.Column(scale=2):
                                gr.Markdown("### Items")

                                wc_items_display = gr.Textbox(
                                    label="Category Items (editable - one per line)",
                                    lines=15,
                                    max_lines=25,
                                    interactive=True,
                                    placeholder="Select a category to view and edit items..."
                                )

                                with gr.Row():
                                    wc_save_items_btn = gr.Button("Save Changes", variant="primary")
                                    wc_sort_items_btn = gr.Button("Sort A-Z", size="sm")

                                with gr.Accordion("Add Items", open=False):
                                    wc_add_items_text = gr.Textbox(
                                        label="New Items (one per line)",
                                        placeholder="Enter items to add...",
                                        lines=4
                                    )
                                    wc_add_items_btn = gr.Button("Add Items", variant="primary")

                                with gr.Accordion("Remove Items", open=False):
                                    wc_remove_items_text = gr.Textbox(
                                        label="Items to Remove (one per line)",
                                        placeholder="Enter items to remove...",
                                        lines=4
                                    )
                                    wc_remove_items_btn = gr.Button("Remove Items", variant="stop")

                        # LLM Generation Section
                        gr.Markdown("---\n### Generate Wildcards with LLM")
                        gr.Markdown("""*Ask the LLM to generate lists for you. Just describe what you want:*
- **"types of metals"** → gold, silver, bronze, copper, iron, steel...
- **"fantasy creatures"** → dragon, phoenix, unicorn, griffin...
- **"glass types"** → stained glass, frosted glass, tempered glass...
- **"cyberpunk clothing"** → neon jacket, holographic visor...""")

                        with gr.Row():
                            with gr.Column(scale=2):
                                wc_llm_prompt = gr.Textbox(
                                    label="Ask the LLM (e.g., 'give me all types of metals')",
                                    placeholder="types of wood, fantasy weapons, art movements, weather conditions...",
                                    lines=2
                                )
                            with gr.Column(scale=1):
                                wc_llm_category = gr.Textbox(
                                    label="Category Name (auto-suggested)",
                                    placeholder="e.g., material-metal"
                                )

                        with gr.Row():
                            wc_llm_model = gr.Dropdown(
                                label="Ollama Model",
                                choices=get_ollama_models_list() if ollama_available else OLLAMA_MODELS,
                                value="qwen2.5:7b-instruct"
                            )
                            wc_llm_count = gr.Slider(
                                label="Number of items",
                                minimum=10,
                                maximum=200,
                                value=50,
                                step=10
                            )
                            wc_llm_add_existing = gr.Checkbox(
                                label="Add to existing category",
                                value=True,
                                info="Uncheck to replace"
                            )

                        with gr.Row():
                            wc_llm_generate_btn = gr.Button("Generate with LLM", variant="primary", size="lg")

                        wc_llm_preview = gr.Textbox(
                            label="Generated Items Preview",
                            lines=8,
                            interactive=False,
                            placeholder="Generated items will appear here..."
                        )

                        # Search and Stats
                        gr.Markdown("---")
                        with gr.Row():
                            with gr.Column(scale=1):
                                wc_search_input = gr.Textbox(
                                    label="Search Items",
                                    placeholder="Search across all categories..."
                                )
                                wc_search_btn = gr.Button("Search", size="sm")
                            with gr.Column(scale=1):
                                with gr.Accordion("Backups", open=False):
                                    wc_backup_dropdown = gr.Dropdown(
                                        label="Restore from Backup",
                                        choices=we.get_backup_list()
                                    )
                                    wc_restore_btn = gr.Button("Restore Backup", variant="secondary", size="sm")

                        wc_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2
                        )

                        wc_search_results = gr.Markdown("")
                    else:
                        gr.Markdown("**Wildcard Editor not available.** Make sure wildcard_editor.py is in the project directory.")

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
            fn=refresh_ollama_status_and_models,
            outputs=[ollama_status, ollama_model, batch_ollama_model, wc_llm_model]
        )

        start_ollama_btn.click(
            fn=start_ollama_server,
            outputs=[ollama_status, ollama_model, batch_ollama_model, wc_llm_model]
        )

        stop_ollama_btn.click(
            fn=stop_ollama_server,
            outputs=[ollama_status]
        )

        pull_model_btn.click(
            fn=pull_ollama_model,
            inputs=[new_model_name],
            outputs=[model_action_status, ollama_model, batch_ollama_model, wc_llm_model]
        )

        delete_model_btn.click(
            fn=delete_ollama_model,
            inputs=[ollama_model],
            outputs=[model_action_status, ollama_model, batch_ollama_model, wc_llm_model]
        )

        # Model load/unload handlers
        def load_model_with_status():
            """Wrapper for load_model that updates markdown status."""
            for status in load_model():
                yield status + "\n\n" + get_model_status()
            yield get_model_status()

        def unload_model_with_status():
            """Wrapper for unload_model that updates markdown status."""
            result = unload_model()
            return result + "\n\n" + get_model_status()

        load_model_btn.click(
            fn=load_model_with_status,
            outputs=[model_status]
        )

        unload_model_btn.click(
            fn=unload_model_with_status,
            outputs=[model_status]
        )

        refresh_status_btn.click(
            fn=get_model_status,
            outputs=[model_status]
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
                guidance_scale,
            ],
            outputs=[output_image, status_text, info_text, last_seed],
        )

        reroll_btn.click(
            fn=reroll_with_new_seed,
            inputs=[
                prompt,
                negative_prompt,
                style,
                aspect_ratio,
                custom_size,
                quality_preset,
                custom_steps,
                use_custom_steps,
                save_image,
                use_ollama,
                ollama_model,
                ollama_length,
                ollama_complexity,
                guidance_scale,
            ],
            outputs=[output_image, status_text, info_text, last_seed],
        )

        # Stop generation button
        stop_generation_btn.click(
            fn=stop_single_generation,
            outputs=[status_text]
        )

        # Save/Load prompt config handlers
        save_prompt_config_btn.click(
            fn=lambda *args: save_prompt_config(*args, overwrite=False),
            inputs=[
                prompt_config_name,
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
                use_ollama,
                ollama_model,
                ollama_length,
                ollama_complexity,
                batch_count,
            ],
            outputs=[prompt_config_status, prompt_config_dropdown]
        )

        overwrite_prompt_config_btn.click(
            fn=lambda *args: save_prompt_config(*args, overwrite=True),
            inputs=[
                prompt_config_name,
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
                use_ollama,
                ollama_model,
                ollama_length,
                ollama_complexity,
                batch_count,
            ],
            outputs=[prompt_config_status, prompt_config_dropdown]
        )

        load_prompt_config_btn.click(
            fn=load_prompt_config,
            inputs=[prompt_config_dropdown],
            outputs=[
                prompt_config_name,
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
                use_ollama,
                ollama_model,
                ollama_length,
                ollama_complexity,
                batch_count,
                prompt_config_status
            ]
        )

        delete_prompt_config_btn.click(
            fn=delete_prompt_config,
            inputs=[prompt_config_dropdown],
            outputs=[prompt_config_status, prompt_config_dropdown]
        )

        refresh_prompt_configs_btn.click(
            fn=lambda: gr.update(choices=get_saved_prompt_configs()),
            outputs=[prompt_config_dropdown]
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
                negative_prompt,
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

        # Generate prompts only (no rendering)
        generate_prompts_only_btn.click(
            fn=generate_prompts_only,
            inputs=[
                batch_name,
                batch_themes,
                batch_variations,
                batch_styles,
                batch_images_per,
                batch_llm_backend,
                batch_ollama_model,
                batch_enhance,
                batch_aspect,
                batch_quality,
                batch_random_seeds,
                batch_negative_prompt,
                batch_ollama_length,
                batch_ollama_complexity,
                batch_guidance_scale,
            ],
            outputs=[batch_status]
        )

        # Toggle LLM backend UI (Ollama vs LM Studio)
        def toggle_llm_backend(backend):
            if backend == "LM Studio":
                # Check if LM Studio is available and get loaded model
                if lmstudio_available and lms.check_lmstudio_available():
                    model_name = lms.get_lmstudio_model()
                    status = f"Connected: {model_name}" if model_name else "Connected (no model info)"
                else:
                    status = "Not running - start LM Studio server"
                return gr.update(visible=False), gr.update(visible=True, value=status)
            else:
                return gr.update(visible=True), gr.update(visible=False)

        batch_llm_backend.change(
            fn=toggle_llm_backend,
            inputs=[batch_llm_backend],
            outputs=[batch_ollama_model, batch_lmstudio_status]
        )

        batch_start_btn.click(
            fn=run_batch_generation,
            inputs=[
                batch_themes,
                batch_variations,
                batch_styles,
                batch_images_per,
                batch_llm_backend,
                batch_ollama_model,
                batch_enhance,
                batch_aspect,
                batch_quality,
                batch_random_seeds,
                batch_name,
                batch_ollama_length,
                batch_ollama_complexity,
                batch_negative_prompt,
                batch_guidance_scale,
            ],
            outputs=[gr.State(), batch_status, batch_output_dir, batch_gallery]
        )

        batch_stop_btn.click(
            fn=stop_batch,
            outputs=[batch_status]
        )

        # ============================================================
        # SAVED PROMPT RUNS HANDLERS
        # ============================================================

        # Preview when selecting a prompt run
        prompt_run_dropdown.change(
            fn=preview_prompt_run,
            inputs=[prompt_run_dropdown],
            outputs=[prompt_run_preview]
        )

        # Refresh prompt runs list
        refresh_prompt_runs_btn.click(
            fn=lambda: gr.update(choices=get_saved_prompt_runs()),
            outputs=[prompt_run_dropdown]
        )

        # Run from saved prompt run
        run_from_saved_btn.click(
            fn=run_from_prompt_run,
            inputs=[prompt_run_dropdown, interleave_checkbox, start_at_prompt, loop_back_checkbox],
            outputs=[prompt_run_gallery, prompt_run_status, prompt_run_output_dir, prompt_run_gallery]
        )

        # Stop prompt run (uses same stop as batch)
        stop_prompt_run_btn.click(
            fn=stop_batch,
            outputs=[prompt_run_status]
        )

        # Delete prompt run
        delete_prompt_run_btn.click(
            fn=delete_prompt_run,
            inputs=[prompt_run_dropdown],
            outputs=[prompt_run_status, prompt_run_dropdown]
        )

        # ============================================================
        # COMBINE CONFIGS HANDLERS
        # ============================================================

        # Preview uploaded configs
        combine_file_upload.change(
            fn=preview_combined_configs,
            inputs=[combine_file_upload],
            outputs=[combine_preview]
        )

        # Create prompt run from configs
        def handle_combine_configs(files, run_name, images_per, use_seeds):
            success, message, dropdown_update = combine_configs_to_prompt_run(
                files, run_name, int(images_per), use_seeds
            )
            return message, dropdown_update

        combine_create_btn.click(
            fn=handle_combine_configs,
            inputs=[combine_file_upload, combine_run_name, combine_images_per, combine_use_seeds],
            outputs=[combine_status, prompt_run_dropdown]
        )

        # Clear uploaded files
        combine_clear_btn.click(
            fn=lambda: (None, "Upload JSON config files to preview...", ""),
            outputs=[combine_file_upload, combine_preview, combine_status]
        )

        # Auto-calculate batch on setting changes
        for component in [batch_themes, batch_variations, batch_styles, batch_images_per]:
            component.change(
                fn=calculate_batch_total,
                inputs=[batch_themes, batch_variations, batch_styles, batch_images_per],
                outputs=[batch_preview]
            )

        # ============================================================
        # BATCH CONFIG SAVE/LOAD HANDLERS (Enhanced)
        # ============================================================

        if batch_manager_available:
            # Save with custom name
            batch_save_btn.click(
                fn=bm.save_batch_config_named,
                inputs=[
                    config_name_input,
                    batch_name,
                    batch_themes,
                    batch_variations,
                    batch_styles,
                    batch_images_per,
                    batch_llm_backend,
                    batch_ollama_model,
                    batch_enhance,
                    batch_aspect,
                    batch_quality,
                    batch_random_seeds,
                    batch_negative_prompt,
                    batch_ollama_length,
                    batch_ollama_complexity
                ],
                outputs=[batch_config_status, batch_config_dropdown]
            )

            # Overwrite selected config
            def overwrite_config(selected_config, batch_name, themes, variations, styles, images_per,
                                llm_backend, ollama_model, enhance, aspect, quality, random_seeds, negative,
                                ollama_length, ollama_complexity):
                if not selected_config:
                    return "No config selected to overwrite", gr.update()
                config_name = selected_config.replace('.json', '')
                return bm.save_batch_config_named(
                    config_name, batch_name, themes, variations, styles, images_per,
                    llm_backend, ollama_model, enhance, aspect, quality, random_seeds, negative,
                    ollama_length, ollama_complexity
                )

            batch_save_overwrite_btn.click(
                fn=overwrite_config,
                inputs=[
                    batch_config_dropdown,
                    batch_name,
                    batch_themes,
                    batch_variations,
                    batch_styles,
                    batch_images_per,
                    batch_llm_backend,
                    batch_ollama_model,
                    batch_enhance,
                    batch_aspect,
                    batch_quality,
                    batch_random_seeds,
                    batch_negative_prompt,
                    batch_ollama_length,
                    batch_ollama_complexity
                ],
                outputs=[batch_config_status, batch_config_dropdown]
            )

            # Load config with full fields
            batch_load_btn.click(
                fn=bm.load_batch_config_full,
                inputs=[batch_config_dropdown],
                outputs=[
                    batch_name,
                    batch_themes,
                    batch_variations,
                    batch_styles,
                    batch_images_per,
                    batch_llm_backend,
                    batch_ollama_model,
                    batch_enhance,
                    batch_aspect,
                    batch_quality,
                    batch_random_seeds,
                    batch_negative_prompt,
                    batch_ollama_length,
                    batch_ollama_complexity,
                    config_name_input,
                    batch_config_status
                ]
            )

            # Delete config
            batch_delete_config_btn.click(
                fn=bm.delete_batch_config,
                inputs=[batch_config_dropdown],
                outputs=[batch_config_status, batch_config_dropdown]
            )

            # Rename config
            batch_rename_btn.click(
                fn=bm.rename_batch_config,
                inputs=[batch_config_dropdown, batch_rename_input],
                outputs=[batch_config_status, batch_config_dropdown]
            )

            # Refresh configs list
            batch_refresh_configs_btn.click(
                fn=lambda: gr.update(choices=bm.get_config_choices()),
                outputs=[batch_config_dropdown]
            )
        else:
            # Fallback to original handlers if batch_manager not available
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

        # Custom directory browser - from file picker
        browse_from_file_btn.click(
            fn=load_batch_from_file,
            inputs=[browse_file_picker],
            outputs=[browse_batch_dropdown, custom_batch_dir, custom_browse_status]
        )

        # Custom directory browser - from text path
        browse_custom_btn.click(
            fn=load_custom_batch_directory,
            inputs=[custom_batch_dir],
            outputs=[browse_batch_dropdown, custom_browse_status]
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

        # Click on image in browse gallery - show info and config for editing
        def select_image_for_editing(evt: gr.SelectData, current_batch: str):
            """Select an image and load its info/config for editing"""
            global custom_batch_base_dir

            if evt is None or evt.value is None:
                return None, "No image selected", "{}", "No image selected"

            # Get image path
            if isinstance(evt.value, dict):
                image_path = evt.value.get('image', {}).get('path', '')
            elif isinstance(evt.value, str):
                image_path = evt.value
            else:
                image_path = str(evt.value)

            if not image_path:
                return None, "Could not get image path", "{}", "Error getting path"

            # Find the actual config file
            config_path = Path(image_path).with_suffix('.json')

            # If not found at direct path, try batch directory
            if not config_path.exists() and current_batch and current_batch != "(No batches yet)":
                filename = Path(image_path).stem
                base_dir = Path(custom_batch_base_dir) if custom_batch_base_dir else OUTPUT_DIR / "batches"
                batch_dir = base_dir / current_batch
                config_path = batch_dir / f"{filename}.json"
                # Also update image_path to the actual path
                actual_image = batch_dir / f"{filename}.png"
                if actual_image.exists():
                    image_path = str(actual_image)

            # Load config
            if batch_manager_available:
                info_text = bm.get_image_info_display(image_path)
                config_json = bm.get_image_config_json(image_path)
            else:
                info_text = f"Image: {Path(image_path).name}"
                config_json = "{}"

            return image_path, info_text, config_json, f"Selected: {Path(image_path).name}"

        browse_gallery.select(
            fn=select_image_for_editing,
            inputs=[browse_batch_dropdown],
            outputs=[
                selected_image_path,
                selected_image_info,
                selected_image_config,
                browse_load_status
            ]
        )

        # Use selected image settings in main generator
        def use_image_in_generator(image_path: str, current_batch: str):
            """Load image config into main generator UI"""
            if not image_path:
                return [gr.update()] * 9 + ["No image selected"]

            # Use the existing load function
            return load_image_from_gallery_by_path(image_path, current_batch)

        def load_image_from_gallery_by_path(image_path: str, current_batch: str):
            """Load image config by path"""
            global custom_batch_base_dir

            if not image_path:
                return [gr.update()] * 9 + ["No image selected"]

            config_path = Path(image_path).with_suffix('.json')

            # Try to find config in batch directory if not at direct path
            if not config_path.exists() and current_batch and current_batch != "(No batches yet)":
                filename = Path(image_path).stem
                base_dir = Path(custom_batch_base_dir) if custom_batch_base_dir else OUTPUT_DIR / "batches"
                batch_dir = base_dir / current_batch
                config_path = batch_dir / f"{filename}.json"

            if not config_path.exists():
                return [gr.update()] * 9 + [f"No config found for: {Path(image_path).name}"]

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
                    gr.update(value=False),
                    gr.update(value=config.get('use_ollama', False)),
                    gr.update(value=config.get('negative_prompt', '')),
                    f"Loaded: {Path(image_path).name}\nSeed: {config.get('seed')}"
                ]
            except Exception as e:
                return [gr.update()] * 9 + [f"Error loading config: {e}"]

        browse_use_settings_btn.click(
            fn=use_image_in_generator,
            inputs=[selected_image_path, browse_batch_dropdown],
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

        # Save edited config
        def save_edited_config(image_path: str, config_json: str):
            """Save edited JSON config back to file"""
            if not image_path:
                return "No image selected"
            if batch_manager_available:
                return bm.update_image_config_from_json(image_path, config_json)
            return "Batch manager not available"

        browse_save_config_btn.click(
            fn=save_edited_config,
            inputs=[selected_image_path, selected_image_config],
            outputs=[browse_load_status]
        )

        # Reset config (reload from file)
        def reset_config_display(image_path: str):
            """Reload config from file"""
            if not image_path:
                return "{}", "No image selected"
            if batch_manager_available:
                return bm.get_image_config_json(image_path), "Config reset from file"
            return "{}", "Batch manager not available"

        browse_reset_config_btn.click(
            fn=reset_config_display,
            inputs=[selected_image_path],
            outputs=[selected_image_config, browse_load_status]
        )

        # Delete individual image
        def delete_single_image(image_path: str, current_batch: str, current_page: int):
            """Delete a single image and refresh gallery"""
            if not image_path:
                return gr.update(), "Select a batch...", 0, 0, "Page 1 of 1", "No image selected", "", "{}"

            if batch_manager_available:
                msg, success = bm.delete_image(image_path)
            else:
                # Fallback delete
                try:
                    Path(image_path).unlink(missing_ok=True)
                    Path(image_path).with_suffix('.json').unlink(missing_ok=True)
                    msg, success = "Deleted", True
                except Exception as e:
                    msg, success = f"Error: {e}", False

            if success:
                # Refresh the gallery
                images, info, page, total_pages = get_batch_images(current_batch, current_page)
                page_text = f"Page {page + 1} of {total_pages}"
                return images, info, page, total_pages, page_text, msg, "Click an image to select...", "{}"
            else:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), msg, gr.update(), gr.update()

        browse_delete_image_btn.click(
            fn=delete_single_image,
            inputs=[selected_image_path, browse_batch_dropdown, browse_page_state],
            outputs=[browse_gallery, browse_info, browse_page_state, browse_total_pages, browse_page_info,
                    browse_load_status, selected_image_info, selected_image_config]
        )

        # Delete entire batch
        def delete_entire_batch(batch_name: str):
            """Delete entire batch directory"""
            global custom_batch_base_dir

            if not batch_name or batch_name == "(No batches yet)":
                return gr.update(), "No batch selected", [], "Select a batch...", 0, 0, "Page 1 of 1"

            if batch_manager_available:
                msg, dropdown_update = bm.delete_batch_directory(batch_name, custom_batch_base_dir)
            else:
                # Fallback delete
                try:
                    base_dir = Path(custom_batch_base_dir) if custom_batch_base_dir else OUTPUT_DIR / "batches"
                    import shutil
                    shutil.rmtree(base_dir / batch_name)
                    dirs = get_batch_directories()
                    dropdown_update = gr.update(choices=dirs, value=dirs[0] if dirs else None)
                    msg = f"Deleted: {batch_name}"
                except Exception as e:
                    dropdown_update = gr.update()
                    msg = f"Error: {e}"

            return dropdown_update, msg, [], "Select a batch...", 0, 0, "Page 1 of 1"

        browse_delete_batch_btn.click(
            fn=delete_entire_batch,
            inputs=[browse_batch_dropdown],
            outputs=[browse_batch_dropdown, browse_load_status, browse_gallery, browse_info,
                    browse_page_state, browse_total_pages, browse_page_info]
        )

        # Load batch settings into Configure tab
        def load_batch_settings_for_rerun(batch_name: str):
            """Load batch manifest settings into UI"""
            global custom_batch_base_dir

            if not batch_name or batch_name == "(No batches yet)":
                return [gr.update()] * 14 + ["No batch selected"]

            if batch_manager_available:
                return bm.load_batch_settings_to_ui(batch_name, custom_batch_base_dir)
            else:
                # Fallback: try to load manifest directly
                base_dir = Path(custom_batch_base_dir) if custom_batch_base_dir else OUTPUT_DIR / "batches"
                manifest_path = base_dir / batch_name / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                        settings = manifest.get('settings', {})
                        themes = settings.get('themes', [])
                        themes_text = '\n'.join(themes) if isinstance(themes, list) else themes
                        return [
                            gr.update(value=manifest.get('batch_name', batch_name)),
                            gr.update(value=themes_text),
                            gr.update(value=settings.get('variations_per_theme', 3)),
                            gr.update(value=settings.get('styles', [])),
                            gr.update(value=settings.get('images_per_combo', 1)),
                            gr.update(value=settings.get('llm_backend', 'Ollama')),
                            gr.update(value=settings.get('ollama_model', 'qwen2.5:7b-instruct')),
                            gr.update(value=settings.get('enhance_prompts', True)),
                            gr.update(value=settings.get('aspect_ratio', '1:1 (Square)')),
                            gr.update(value=settings.get('quality', 'Standard')),
                            gr.update(value=True),
                            gr.update(value=settings.get('negative_prompt', '')),
                            gr.update(value=settings.get('ollama_length', 'medium')),
                            gr.update(value=settings.get('ollama_complexity', 'detailed')),
                            f"Loaded from: {batch_name}"
                        ]
                    except Exception as e:
                        return [gr.update()] * 14 + [f"Error: {e}"]
                return [gr.update()] * 14 + ["No manifest found"]

        browse_load_settings_btn.click(
            fn=load_batch_settings_for_rerun,
            inputs=[browse_batch_dropdown],
            outputs=[
                batch_name,
                batch_themes,
                batch_variations,
                batch_styles,
                batch_images_per,
                batch_llm_backend,
                batch_ollama_model,
                batch_enhance,
                batch_aspect,
                batch_quality,
                batch_random_seeds,
                batch_negative_prompt,
                batch_ollama_length,
                batch_ollama_complexity,
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

        # ============================================================
        # STYLE EDITOR HANDLERS
        # ============================================================

        # Load style details when selected
        def load_style_for_editing(style_name):
            if not style_name or style_name == "None":
                return "", ""
            suffix = STYLE_PRESETS.get(style_name, "")
            return style_name, suffix

        style_editor_dropdown.change(
            fn=load_style_for_editing,
            inputs=[style_editor_dropdown],
            outputs=[style_name_input, style_suffix_input]
        )

        # Refresh style dropdown
        style_refresh_btn.click(
            fn=lambda: gr.update(choices=list(STYLE_PRESETS.keys())),
            outputs=[style_editor_dropdown]
        )

        # Add/Update style - update all style dropdowns
        def handle_add_style(name, suffix):
            status, dropdown_update = add_style_preset(name, suffix)
            # Return updates for all style-related dropdowns
            choices = list(STYLE_PRESETS.keys())
            return (
                status,
                gr.update(choices=choices, value=name),  # style_editor_dropdown
                gr.update(choices=choices),  # main style dropdown
                gr.update(choices=choices),  # batch_styles (CheckboxGroup)
            )

        style_add_btn.click(
            fn=handle_add_style,
            inputs=[style_name_input, style_suffix_input],
            outputs=[style_editor_status, style_editor_dropdown, style, batch_styles]
        )

        # Delete style - update all style dropdowns
        def handle_delete_style(name):
            status, dropdown_update = delete_style_preset(name)
            choices = list(STYLE_PRESETS.keys())
            return (
                status,
                gr.update(choices=choices, value="None"),  # style_editor_dropdown
                gr.update(choices=choices, value="None"),  # main style dropdown
                gr.update(choices=choices),  # batch_styles
                "", ""  # Clear name and suffix inputs
            )

        style_delete_btn.click(
            fn=handle_delete_style,
            inputs=[style_editor_dropdown],
            outputs=[style_editor_status, style_editor_dropdown, style, batch_styles, style_name_input, style_suffix_input]
        )

        # Reset to defaults - update all style dropdowns
        def handle_reset_styles():
            status, dropdown_update = reset_style_presets()
            choices = list(STYLE_PRESETS.keys())
            return (
                status,
                gr.update(choices=choices, value="None"),  # style_editor_dropdown
                gr.update(choices=choices, value="None"),  # main style dropdown
                gr.update(choices=choices),  # batch_styles
                "", ""  # Clear inputs
            )

        style_reset_btn.click(
            fn=handle_reset_styles,
            outputs=[style_editor_status, style_editor_dropdown, style, batch_styles, style_name_input, style_suffix_input]
        )

        # Export styles to text
        style_export_btn.click(
            fn=export_styles_to_text,
            outputs=[style_bulk_text]
        )

        # Import styles from text - update all style dropdowns
        def handle_import_styles(text):
            status, dropdown_update = import_styles_from_text(text)
            choices = list(STYLE_PRESETS.keys())
            return (
                status,
                gr.update(choices=choices, value="None"),  # style_editor_dropdown
                gr.update(choices=choices, value="None"),  # main style dropdown
                gr.update(choices=choices),  # batch_styles
            )

        style_import_btn.click(
            fn=handle_import_styles,
            inputs=[style_bulk_text],
            outputs=[style_editor_status, style_editor_dropdown, style, batch_styles]
        )

        # ============================================================
        # WILDCARD EDITOR HANDLERS
        # ============================================================

        if wildcard_editor_available:

            # Helper to refresh main wildcard dropdown after edits
            def get_main_wildcard_update():
                """Get update for main wildcard dropdown after changes"""
                if wildcard_manager:
                    wildcard_manager.reload()
                    return gr.update(choices=wildcard_manager.get_available_wildcards())
                return gr.update()

            # Load category items when selected
            def load_wc_category(category):
                if not category:
                    return "", "Select a category to view items..."
                items = we.get_category_items(category)
                info = f"**{category}**: {len(items)} items"
                return "\n".join(sorted(items)), info

            wc_category_dropdown.change(
                fn=load_wc_category,
                inputs=[wc_category_dropdown],
                outputs=[wc_items_display, wc_category_info]
            )

            # Refresh categories list (also refreshes main dropdown)
            def refresh_wc_categories():
                cats = we.get_category_list()
                main_update = get_main_wildcard_update()
                return gr.update(choices=cats), main_update

            wc_refresh_btn.click(
                fn=refresh_wc_categories,
                outputs=[wc_category_dropdown, wildcard_dropdown] if wildcard_available else [wc_category_dropdown]
            )

            # Show stats
            wc_stats_btn.click(
                fn=we.get_stats,
                outputs=[wc_search_results]
            )

            # Add new category
            def add_wc_category(name, items):
                msg, dropdown_update = we.add_category(name, items)
                main_update = get_main_wildcard_update()
                return msg, dropdown_update, "", "", main_update

            wc_add_category_btn.click(
                fn=add_wc_category,
                inputs=[wc_new_category_name, wc_new_category_items],
                outputs=[wc_status, wc_category_dropdown, wc_new_category_name, wc_new_category_items, wildcard_dropdown] if wildcard_available else [wc_status, wc_category_dropdown, wc_new_category_name, wc_new_category_items]
            )

            # Rename category
            def rename_wc_category(old_name, new_name):
                msg, dropdown_update = we.rename_category(old_name, new_name)
                main_update = get_main_wildcard_update()
                return msg, dropdown_update, "", main_update

            wc_rename_btn.click(
                fn=rename_wc_category,
                inputs=[wc_category_dropdown, wc_rename_input],
                outputs=[wc_status, wc_category_dropdown, wc_rename_input, wildcard_dropdown] if wildcard_available else [wc_status, wc_category_dropdown, wc_rename_input]
            )

            # Delete category
            def delete_wc_category(category):
                msg, dropdown_update, items = we.delete_category(category)
                main_update = get_main_wildcard_update()
                return msg, dropdown_update, items, "Select a category...", main_update

            wc_delete_category_btn.click(
                fn=delete_wc_category,
                inputs=[wc_category_dropdown],
                outputs=[wc_status, wc_category_dropdown, wc_items_display, wc_category_info, wildcard_dropdown] if wildcard_available else [wc_status, wc_category_dropdown, wc_items_display, wc_category_info]
            )

            # Save items changes
            def save_wc_items(category, items_text):
                msg, new_items = we.replace_category_items(category, items_text)
                main_update = get_main_wildcard_update()
                return msg, new_items, main_update

            wc_save_items_btn.click(
                fn=save_wc_items,
                inputs=[wc_category_dropdown, wc_items_display],
                outputs=[wc_status, wc_items_display, wildcard_dropdown] if wildcard_available else [wc_status, wc_items_display]
            )

            # Sort items
            def sort_wc_items(items_text):
                if not items_text:
                    return ""
                items = [i.strip() for i in items_text.split('\n') if i.strip()]
                return "\n".join(sorted(items))

            wc_sort_items_btn.click(
                fn=sort_wc_items,
                inputs=[wc_items_display],
                outputs=[wc_items_display]
            )

            # Add items to category
            def add_wc_items(category, new_items):
                msg, updated_items = we.add_items_to_category(category, new_items)
                main_update = get_main_wildcard_update()
                return msg, updated_items, "", main_update

            wc_add_items_btn.click(
                fn=add_wc_items,
                inputs=[wc_category_dropdown, wc_add_items_text],
                outputs=[wc_status, wc_items_display, wc_add_items_text, wildcard_dropdown] if wildcard_available else [wc_status, wc_items_display, wc_add_items_text]
            )

            # Remove items from category
            def remove_wc_items(category, items_to_remove):
                msg, updated_items = we.remove_items_from_category(category, items_to_remove)
                main_update = get_main_wildcard_update()
                return msg, updated_items, "", main_update

            wc_remove_items_btn.click(
                fn=remove_wc_items,
                inputs=[wc_category_dropdown, wc_remove_items_text],
                outputs=[wc_status, wc_items_display, wc_remove_items_text, wildcard_dropdown] if wildcard_available else [wc_status, wc_items_display, wc_remove_items_text]
            )

            # Auto-suggest category name when LLM prompt changes
            wc_llm_prompt.change(
                fn=we.suggest_category_name,
                inputs=[wc_llm_prompt],
                outputs=[wc_llm_category]
            )

            # Generate wildcards with LLM
            def generate_wc_with_llm(prompt, category, model, count, add_existing):
                msg, preview, dropdown_update = we.generate_wildcards_with_llm(
                    prompt, category, model, int(count), add_existing
                )
                main_update = get_main_wildcard_update()
                return msg, preview, dropdown_update, main_update

            wc_llm_generate_btn.click(
                fn=generate_wc_with_llm,
                inputs=[wc_llm_prompt, wc_llm_category, wc_llm_model, wc_llm_count, wc_llm_add_existing],
                outputs=[wc_status, wc_llm_preview, wc_category_dropdown, wildcard_dropdown] if wildcard_available else [wc_status, wc_llm_preview, wc_category_dropdown]
            )

            # Search items
            wc_search_btn.click(
                fn=we.search_items_across_categories,
                inputs=[wc_search_input],
                outputs=[wc_search_results]
            )

            # Restore from backup
            def restore_wc_backup(backup_name):
                msg, dropdown_update = we.restore_from_backup(backup_name)
                main_update = get_main_wildcard_update()
                return msg, dropdown_update, main_update

            wc_restore_btn.click(
                fn=restore_wc_backup,
                inputs=[wc_backup_dropdown],
                outputs=[wc_status, wc_category_dropdown, wildcard_dropdown] if wildcard_available else [wc_status, wc_category_dropdown]
            )

    return app


def load_model_on_startup():
    """Load the model at startup."""
    global model, model_loaded, model_load_lock

    if model_loaded and model is not None:
        print("Model already loaded.")
        return

    with model_load_lock:
        # Double-check after acquiring lock
        if model_loaded and model is not None:
            print("Model already loaded.")
            return

        # Check if GPU already has significant memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            if mem_used > 30:
                print(f"WARNING: GPU already has {mem_used:.1f}GB allocated. Skipping load.")
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
                local_files_only=True,  # Fully offline - no network calls
            )

            # Model is loaded - set flag FIRST in case tokenizer fails
            model_loaded = True

            # Try to load tokenizer (may fail with local_files_only but model still works)
            try:
                model.load_tokenizer(model_id, local_files_only=True)
            except Exception as tok_err:
                print(f"[LOAD] Tokenizer warning: {tok_err}")
                try:
                    model.load_tokenizer(model_id)
                except Exception:
                    print("[LOAD] Tokenizer fallback failed - model may still work")

            print("Model loaded successfully! (offline mode)")

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

    # Model is NOT auto-loaded anymore!
    # This allows using GPU memory for LLM prompt generation first.
    # Use the "Load Image Model" button in the UI when ready to render.
    print("NOTE: Image model NOT auto-loaded.")
    print("      Load it manually when ready to render images.")
    print("      This leaves GPU free for LLM prompt generation.")

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
