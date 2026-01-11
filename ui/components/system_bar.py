"""
System Bar Component for HunyuanImage-3.0 UI.
Contains GPU selector, model status, and load/unload controls.
Always visible at the top of the interface.
"""

import gradio as gr
from dataclasses import dataclass
from typing import List, Tuple

from ui.state import get_state, detect_gpus, set_gpu
from core.model_manager import load_model, unload_model, get_model_status, is_model_loaded


@dataclass
class SystemBarComponents:
    """Container for system bar UI components."""
    gpu_dropdown: gr.Dropdown
    model_status: gr.Markdown
    load_btn: gr.Button
    unload_btn: gr.Button
    refresh_btn: gr.Button


def get_gpu_choices() -> List[str]:
    """Get list of GPU choices for dropdown."""
    state = get_state()
    if not state.available_gpus:
        state.available_gpus = detect_gpus()
    return [gpu['display'] for gpu in state.available_gpus]


def get_default_gpu() -> str:
    """Get default GPU selection (prefers Blackwell/GPU 1)."""
    choices = get_gpu_choices()
    state = get_state()

    # Find the configured default GPU
    for gpu in state.available_gpus:
        if gpu['index'] == state.selected_gpu:
            return gpu['display']

    # Fallback to first available
    return choices[0] if choices else "No GPU"


def handle_gpu_change(gpu_choice: str) -> str:
    """Handle GPU selection change."""
    state = get_state()

    # Find GPU index from choice string
    for gpu in state.available_gpus:
        if gpu['display'] == gpu_choice:
            gpu_idx = gpu['index']

            # Check if model is loaded
            if is_model_loaded():
                return f"**WARNING**: Model is loaded on GPU {state.selected_gpu}. Unload model first, then change GPU and reload."

            # Apply GPU change
            result = set_gpu(gpu_idx)
            return get_model_status() + f"\n\n*{result}*"

    return "GPU selection failed"


def handle_load_model():
    """Handle model load button click."""
    for status in load_model():
        yield status


def handle_unload_model() -> str:
    """Handle model unload button click."""
    return unload_model()


def handle_refresh_status() -> str:
    """Handle refresh button click."""
    return get_model_status()


def create_system_bar() -> SystemBarComponents:
    """Create the system bar UI component.

    Returns:
        SystemBarComponents with references to all widgets.
    """
    # Initialize GPU detection
    state = get_state()
    if not state.available_gpus:
        state.available_gpus = detect_gpus()

    with gr.Group():
        gr.Markdown("### System")
        with gr.Row():
            # GPU Selection
            gpu_dropdown = gr.Dropdown(
                label="GPU",
                choices=get_gpu_choices(),
                value=get_default_gpu(),
                scale=2,
                info="Select GPU for image generation"
            )

            # Model Status
            model_status = gr.Markdown(
                value=get_model_status(),
                elem_id="model_status"
            )

        with gr.Row():
            # Control buttons
            load_btn = gr.Button(
                "Load Model",
                variant="primary",
                size="sm",
                scale=1
            )
            unload_btn = gr.Button(
                "Unload Model",
                variant="stop",
                size="sm",
                scale=1
            )
            refresh_btn = gr.Button(
                "Refresh",
                variant="secondary",
                size="sm",
                scale=1
            )

    return SystemBarComponents(
        gpu_dropdown=gpu_dropdown,
        model_status=model_status,
        load_btn=load_btn,
        unload_btn=unload_btn,
        refresh_btn=refresh_btn
    )


def wire_system_bar_events(components: SystemBarComponents) -> None:
    """Wire up event handlers for system bar components.

    Args:
        components: SystemBarComponents instance to wire up.
    """
    # GPU selection change
    components.gpu_dropdown.change(
        fn=handle_gpu_change,
        inputs=[components.gpu_dropdown],
        outputs=[components.model_status]
    )

    # Load model button
    components.load_btn.click(
        fn=handle_load_model,
        inputs=[],
        outputs=[components.model_status]
    )

    # Unload model button
    components.unload_btn.click(
        fn=handle_unload_model,
        inputs=[],
        outputs=[components.model_status]
    )

    # Refresh status button
    components.refresh_btn.click(
        fn=handle_refresh_status,
        inputs=[],
        outputs=[components.model_status]
    )
