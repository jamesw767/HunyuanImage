#!/usr/bin/env python3
"""
Test UI for the new modular components.
Run this to validate the new layout before modifying the main hunyuan_ui.py.

Usage:
    source hunyuan_env/bin/activate
    python test_new_ui.py
"""

import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gradio as gr

# Import our modular components
from ui.state import get_state, init_gpus
from ui.constants import OUTPUT_DIR
from ui.components import (
    create_system_bar,
    wire_system_bar_events,
    create_prompt_input,
    wire_prompt_input_events,
    create_gen_settings,
    wire_gen_settings_events,
    create_output_display,
    wire_output_display_events,
    get_steps_from_quality,
    get_size_from_aspect,
    create_batch_panel,
)

# Initialize Ollama if available
ollama_available = False
try:
    from ollama_manager import OllamaManager
    from ollama_prompts import PromptEnhancer
    state = get_state()
    state.ollama_manager = OllamaManager()
    state.ollama_available = True
    ollama_available = True
    print("[INIT] Ollama available")
except ImportError:
    print("[INIT] Ollama not available")


def dummy_generate(prompt, negative, style, aspect, quality, seed, batch_count,
                   use_ollama, ollama_model, ollama_length, ollama_complexity):
    """Dummy generation function for testing UI."""
    steps = get_steps_from_quality(quality)
    size = get_size_from_aspect(aspect)

    info = f"""
**Test Output** (model not loaded)

Prompt: {prompt[:50]}...
Style: {style}
Size: {size}
Steps: {steps}
Seed: {seed}
Batch: {batch_count}
Ollama: {use_ollama} ({ollama_model if use_ollama else 'disabled'})
"""
    return None, "Test complete (no actual generation)", info, seed


def create_test_ui():
    """Create test UI with modular components."""

    with gr.Blocks(
        title="HunyuanImage-3.0 Test UI"
    ) as app:

        # Header
        gr.Markdown("""
        # HunyuanImage-3.0 - New Layout Test
        **Testing modular components** - Ollama controls under prompt, GPU selector at top
        """)

        # System Bar (GPU selector, model controls)
        system = create_system_bar()
        wire_system_bar_events(system)

        # Main layout
        with gr.Row():
            # Left column - inputs
            with gr.Column(scale=1):
                # Prompt + Ollama + Style (grouped together!)
                prompt = create_prompt_input(ollama_available=ollama_available)
                wire_prompt_input_events(prompt)

                # Generation settings
                settings = create_gen_settings()
                wire_gen_settings_events(settings)

            # Right column - output
            with gr.Column(scale=1):
                output = create_output_display()
                wire_output_display_events(output)

        # Wire up generate button
        output.generate_btn.click(
            fn=dummy_generate,
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

        # Batch Panel (opens by default for batch-heavy workflows)
        batch = create_batch_panel(open_by_default=True)

        # Recent gallery placeholder
        with gr.Accordion("Recent Generations", open=False):
            gr.Markdown("Recent gallery placeholder")

    return app


def main():
    print("=" * 50)
    print("HunyuanImage-3.0 TEST UI")
    print("Testing new modular component layout")
    print("=" * 50)

    # Initialize GPUs
    init_gpus()

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nStarting test interface on http://localhost:7861")

    app = create_test_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from main UI
        share=False,
        inbrowser=False
    )


if __name__ == "__main__":
    main()
