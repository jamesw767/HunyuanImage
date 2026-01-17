#!/usr/bin/env python3
"""
HunyuanImage Multi-Engine UI - Version 3

Unified interface for multiple image generation engines:
- HunyuanImage-3.0 (80B MoE) - Best quality at 1024px
- FLUX.2 Klein 9B - Higher resolution support (up to 3072px)

Features:
- Select engine and GPU independently
- Engine-specific resolution presets
- Unified batch processing
- Ollama prompt enhancement (shared)
- Wildcard system (shared)

Usage:
    python hunyuan_ui_v3.py
"""

import os
import sys
import gc
import time
import json
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force offline mode for HuggingFace (Hunyuan)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/media/james/BigDrive/AI/cache/huggingface"

import torch
import gradio as gr
from PIL import Image

# Import engine system
from ui_v3.engines import EngineRegistry, GenerationParams, GenerationResult

# Constants
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize engines on import
print("[V3] Initializing engine registry...")
print(f"[V3] Available engines: {EngineRegistry.get_available_engines()}")


@dataclass
class AppState:
    """Global application state."""
    current_engine_name: str = "hunyuan"
    current_engine: Any = None
    available_gpus: List[Dict] = None
    ollama_available: bool = False
    ollama_enhancer: Any = None
    wildcard_manager: Any = None
    session_dir: Optional[Path] = None
    last_image_path: Optional[str] = None
    last_seed: int = 0

    def __post_init__(self):
        self.available_gpus = []


# Global state
state = AppState()


def detect_gpus() -> List[Dict]:
    """Detect available CUDA GPUs."""
    gpus = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    'index': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'display': f"GPU {i}: {props.name} ({props.total_memory / (1024**3):.0f} GB)"
                })
    except Exception as e:
        print(f"[GPU] Error: {e}")
    return gpus


def get_gpu_status() -> str:
    """Get current GPU status."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    idx, name, temp, util, mem_used, mem_total = parts[:6]
                    lines.append(f"GPU {idx}: {temp}°C | {util}% | {mem_used}/{mem_total} MB")
            return "\n".join(lines)
    except Exception:
        pass
    return "GPU status unavailable"


def init_integrations():
    """Initialize Ollama and Wildcards."""
    # Ollama
    try:
        from ollama_prompts import PromptEnhancer
        state.ollama_available = True
        print("[V3] Ollama available")
    except ImportError:
        state.ollama_available = False
        print("[V3] Ollama not available")

    # Wildcards
    try:
        from wildcard_utils import WildcardManager
        state.wildcard_manager = WildcardManager(json_path=PROJECT_DIR / "wildcards.json")
        print(f"[V3] Wildcards loaded: {len(state.wildcard_manager.data)} categories")
    except Exception as e:
        print(f"[V3] Wildcards not available: {e}")


def get_engine_choices() -> List[str]:
    """Get list of available engines for dropdown."""
    caps = EngineRegistry.get_all_capabilities()
    return [f"{c.display_name}" for c in caps.values()]


def get_engine_name_from_display(display_name: str) -> str:
    """Convert display name back to engine name."""
    caps = EngineRegistry.get_all_capabilities()
    for name, cap in caps.items():
        if cap.display_name == display_name:
            return name
    return "hunyuan"


def get_resolution_presets(engine_name: str) -> List[str]:
    """Get resolution presets for an engine."""
    caps = EngineRegistry.get_all_capabilities()
    if engine_name in caps:
        return list(caps[engine_name].resolution_presets.keys())
    return ["1024x1024"]


def load_engine(engine_display: str, gpu_index: int) -> str:
    """Load the selected engine on the specified GPU."""
    engine_name = get_engine_name_from_display(engine_display)

    # Unload current engine if different
    if state.current_engine is not None:
        if state.current_engine_name != engine_name:
            print(f"[V3] Unloading current engine: {state.current_engine_name}")
            state.current_engine.unload()
            state.current_engine = None

    # Create and load new engine
    print(f"[V3] Loading engine: {engine_name} on GPU {gpu_index}")
    engine = EngineRegistry.get_or_create(engine_name, gpu_index)

    if engine is None:
        return f"Error: Engine '{engine_name}' not found"

    if engine.is_model_loaded():
        state.current_engine = engine
        state.current_engine_name = engine_name
        return f"✓ {engine.capabilities.display_name} already loaded on GPU {gpu_index}"

    success = engine.load()
    if success:
        state.current_engine = engine
        state.current_engine_name = engine_name
        return f"✓ {engine.capabilities.display_name} loaded on GPU {gpu_index}"
    else:
        return f"✗ Failed to load {engine_name}"


def unload_engine() -> str:
    """Unload the current engine."""
    if state.current_engine is None:
        return "No engine loaded"

    name = state.current_engine_name
    state.current_engine.unload()
    state.current_engine = None
    state.current_engine_name = ""
    return f"✓ {name} unloaded"


def update_resolution_dropdown(engine_display: str):
    """Update resolution dropdown when engine changes."""
    engine_name = get_engine_name_from_display(engine_display)
    presets = get_resolution_presets(engine_name)
    caps = EngineRegistry.get_all_capabilities().get(engine_name)

    # Get engine info
    info = ""
    if caps:
        info = f"**{caps.display_name}**\n"
        info += f"- Resolution: {caps.min_width}-{caps.max_width}px\n"
        info += f"- Steps: {caps.min_steps}-{caps.max_steps} (default {caps.default_steps})\n"
        info += f"- VRAM: ~{caps.vram_required_gb:.0f} GB\n"
        if caps.supports_guidance:
            info += f"- Guidance: {caps.min_guidance}-{caps.max_guidance}\n"
        info += f"- Negative prompt: {'Yes' if caps.supports_negative_prompt else 'No'}\n"

    return gr.update(choices=presets, value=presets[0] if presets else "1024x1024"), info


def get_or_create_session_dir(prompt: str = "") -> Path:
    """Get or create session directory for outputs."""
    if state.session_dir is None or not state.session_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in prompt[:25] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        if not safe_name:
            safe_name = "session"
        state.session_dir = OUTPUT_DIR / f"v3_{timestamp}_{safe_name}"
        state.session_dir.mkdir(parents=True, exist_ok=True)
    return state.session_dir


def enhance_prompt(prompt: str, model: str, length: str, complexity: str) -> Tuple[str, str]:
    """Enhance prompt with Ollama."""
    if not state.ollama_available:
        return prompt, "Ollama not available"

    try:
        from ollama_prompts import PromptEnhancer

        if state.ollama_enhancer is None:
            state.ollama_enhancer = PromptEnhancer(model=model)

        enhanced = state.ollama_enhancer.enhance(
            prompt,
            length=length,
            complexity=complexity,
            model=model
        )
        return enhanced, f"Enhanced with {model}"
    except Exception as e:
        return prompt, f"Enhancement failed: {e}"


def generate_image(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    steps: int,
    guidance: float,
    seed: int,
    batch_size: int,
    use_ollama: bool,
    ollama_model: str,
    ollama_length: str,
    ollama_complexity: str,
):
    """Generate image(s) with the current engine."""
    if state.current_engine is None:
        yield None, "No engine loaded. Click 'Load Engine' first.", 0
        return

    if not prompt or not prompt.strip():
        yield None, "Please enter a prompt.", 0
        return

    engine = state.current_engine
    caps = engine.capabilities

    # Parse resolution
    if resolution in caps.resolution_presets:
        width, height = caps.resolution_presets[resolution]
    else:
        # Try to parse "WxH" format
        try:
            width, height = map(int, resolution.lower().split('x'))
        except:
            width, height = caps.default_width, caps.default_height

    # Process prompt
    full_prompt = prompt

    # Ollama enhancement
    if use_ollama and state.ollama_available:
        yield None, f"Enhancing prompt with {ollama_model}...", seed
        full_prompt, status = enhance_prompt(full_prompt, ollama_model, ollama_length, ollama_complexity)

    # Wildcards
    if state.wildcard_manager:
        try:
            full_prompt = state.wildcard_manager.process(full_prompt)
        except:
            pass

    # Build generation params
    params = GenerationParams(
        prompt=full_prompt,
        negative_prompt=negative_prompt if caps.supports_negative_prompt else "",
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance if caps.supports_guidance else 1.0,
        seed=seed,
        batch_size=batch_size,
    )

    yield None, f"Generating {batch_size} image(s) at {width}x{height}...", seed

    # Generate
    result = engine.generate(params)

    if not result.success:
        yield None, f"Generation failed: {result.error_message}", seed
        return

    # Save images
    session_dir = get_or_create_session_dir(prompt)
    last_image = None
    last_seed = seed

    for i, (img, img_seed) in enumerate(zip(result.images, result.seeds)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in " -_").strip().replace(" ", "_")
        filename = f"{timestamp}_{img_seed}_{safe_prompt}.png"
        filepath = session_dir / filename
        img.save(str(filepath))

        # Save config
        config = {
            "prompt": prompt,
            "full_prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "engine": engine.capabilities.name,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance,
            "seed": img_seed,
            "generation_time": result.generation_time,
            "use_ollama": use_ollama,
            "ollama_model": ollama_model if use_ollama else None,
        }
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(config, f, indent=2)

        last_image = str(filepath)
        last_seed = img_seed

        # Update state
        state.last_image_path = last_image
        state.last_seed = last_seed

        yield last_image, f"Generated {i+1}/{batch_size} ({result.generation_time:.1f}s total)", last_seed


def create_ui():
    """Create the v3 multi-engine UI."""

    with gr.Blocks(title="HunyuanImage v3 - Multi-Engine") as app:

        # Header
        gr.Markdown("""
        # HunyuanImage v3 - Multi-Engine Generator
        **Select your engine** | Hunyuan (80B MoE) for quality | FLUX.2 Klein for high-res
        """)

        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):

                # Engine Selection
                gr.Markdown("### Engine Selection")
                with gr.Row():
                    engine_dropdown = gr.Dropdown(
                        label="Engine",
                        choices=get_engine_choices(),
                        value=get_engine_choices()[0] if get_engine_choices() else "HunyuanImage-3.0 (80B MoE)",
                        scale=2
                    )
                    gpu_dropdown = gr.Dropdown(
                        label="GPU",
                        choices=[gpu['display'] for gpu in state.available_gpus],
                        value=state.available_gpus[1]['display'] if len(state.available_gpus) > 1 else (
                            state.available_gpus[0]['display'] if state.available_gpus else "GPU 0"
                        ),
                        scale=1
                    )

                with gr.Row():
                    load_btn = gr.Button("Load Engine", variant="primary")
                    unload_btn = gr.Button("Unload", variant="secondary")

                engine_status = gr.Textbox(label="Engine Status", value="No engine loaded", interactive=False)
                engine_info = gr.Markdown("Select an engine to see details")

                # GPU Status
                gpu_status = gr.Textbox(label="GPU Status", value="", interactive=False, lines=3)
                refresh_gpu_btn = gr.Button("Refresh GPU", size="sm")

                gr.Markdown("---")

                # Prompt
                gr.Markdown("### Prompt")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe your image...",
                    lines=4
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (if supported)",
                    placeholder="What to avoid...",
                    lines=2
                )

                # Ollama Enhancement
                with gr.Accordion("Ollama Enhancement", open=False):
                    use_ollama = gr.Checkbox(label="Enhance with Ollama", value=False)
                    ollama_model = gr.Dropdown(
                        label="Model",
                        choices=["qwen2.5:7b-instruct", "llama3.2:latest"],
                        value="qwen2.5:7b-instruct"
                    )
                    with gr.Row():
                        ollama_length = gr.Dropdown(
                            label="Length",
                            choices=["random", "minimal", "short", "medium", "long", "detailed", "cinematic"],
                            value="medium"
                        )
                        ollama_complexity = gr.Dropdown(
                            label="Complexity",
                            choices=["random", "simple", "moderate", "detailed", "complex", "cinematic"],
                            value="detailed"
                        )

                gr.Markdown("---")

                # Generation Settings
                gr.Markdown("### Generation Settings")
                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=get_resolution_presets("hunyuan"),
                    value="1:1 Square (1024x1024)"
                )
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=10, maximum=100, value=20, step=1)
                    guidance = gr.Slider(label="Guidance (CFG)", minimum=1.0, maximum=15.0, value=4.0, step=0.5)

                with gr.Row():
                    seed = gr.Number(label="Seed (-1=random)", value=-1, precision=0)
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=4, value=1, step=1)

                # Generate Button
                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            # Right column - Output
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image", type="filepath", height=512)
                status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
                last_seed_display = gr.Number(label="Last Seed", value=0, precision=0, interactive=False)

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", size="sm")
                    random_seed_btn = gr.Button("Random Seed", size="sm")

                # Recent Gallery
                with gr.Accordion("Recent Generations", open=False):
                    recent_gallery = gr.Gallery(columns=4, rows=2, height=200)

        # Wire events
        engine_dropdown.change(
            fn=update_resolution_dropdown,
            inputs=[engine_dropdown],
            outputs=[resolution, engine_info]
        )

        def get_gpu_index(gpu_display):
            for gpu in state.available_gpus:
                if gpu['display'] == gpu_display:
                    return gpu['index']
            return 0

        load_btn.click(
            fn=lambda e, g: load_engine(e, get_gpu_index(g)),
            inputs=[engine_dropdown, gpu_dropdown],
            outputs=[engine_status]
        )

        unload_btn.click(
            fn=unload_engine,
            outputs=[engine_status]
        )

        refresh_gpu_btn.click(
            fn=get_gpu_status,
            outputs=[gpu_status]
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt, negative_prompt, resolution, steps, guidance, seed, batch_size,
                use_ollama, ollama_model, ollama_length, ollama_complexity
            ],
            outputs=[output_image, status_text, last_seed_display]
        )

        random_seed_btn.click(
            fn=lambda: random.randint(0, 2**31 - 1),
            outputs=[seed]
        )

        def refresh_latest():
            if state.last_image_path and Path(state.last_image_path).exists():
                return state.last_image_path, "Refreshed", state.last_seed
            return None, "No recent image", 0

        refresh_btn.click(
            fn=refresh_latest,
            outputs=[output_image, status_text, last_seed_display]
        )

        # Initial GPU status
        app.load(fn=get_gpu_status, outputs=[gpu_status])

    return app


def main():
    print("=" * 60)
    print("HunyuanImage v3 - Multi-Engine Generator")
    print("=" * 60)

    # Detect GPUs
    state.available_gpus = detect_gpus()
    print(f"\nGPUs detected: {len(state.available_gpus)}")
    for gpu in state.available_gpus:
        print(f"  {gpu['display']}")

    # Initialize integrations
    init_integrations()

    # Show available engines
    caps = EngineRegistry.get_all_capabilities()
    print(f"\nAvailable engines: {len(caps)}")
    for name, cap in caps.items():
        print(f"  - {cap.display_name}: {cap.min_width}-{cap.max_width}px, ~{cap.vram_required_gb:.0f}GB VRAM")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nStarting web interface on http://localhost:7861")

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from v2
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
