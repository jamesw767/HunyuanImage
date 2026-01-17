"""
Prompt Input Component for HunyuanImage-3.0 UI.
Groups prompt entry, Ollama enhancement, and style selection together.
"""

import gradio as gr
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ui.state import get_state
from ui.constants import (
    DEFAULT_STYLE_PRESETS,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_LENGTH_OPTIONS,
    OLLAMA_COMPLEXITY_OPTIONS,
    OLLAMA_MODELS,
)


@dataclass
class PromptInputComponents:
    """Container for prompt input UI components."""
    # Main prompt
    prompt: gr.Textbox
    negative_prompt: gr.Textbox

    # Ollama enhancement (directly under prompt)
    use_ollama: gr.Checkbox
    ollama_model: gr.Dropdown
    ollama_length: gr.Dropdown
    ollama_complexity: gr.Dropdown

    # Style selection
    style_dropdown: gr.Dropdown

    # Settings loader
    settings_file: gr.File
    load_settings_btn: gr.Button


def get_ollama_models_list() -> List[str]:
    """Get list of available Ollama models."""
    state = get_state()
    if state.ollama_available and state.ollama_manager:
        try:
            models = state.ollama_manager.list_models()
            if models:
                return models
        except Exception:
            pass
    return OLLAMA_MODELS


def get_style_presets() -> dict:
    """Get current style presets."""
    state = get_state()
    if state.style_presets:
        return state.style_presets
    return DEFAULT_STYLE_PRESETS


def create_prompt_input(
    ollama_available: bool = False,
    wildcard_available: bool = False
) -> PromptInputComponents:
    """Create the prompt input UI component.

    Args:
        ollama_available: Whether Ollama is available for enhancement
        wildcard_available: Whether wildcard system is available

    Returns:
        PromptInputComponents with references to all widgets.
    """
    # Main prompt
    gr.Markdown("### Prompt")
    prompt = gr.Textbox(
        label="Describe your image",
        placeholder="A stunning landscape with mountains, a crystal clear lake reflecting the sky...",
        lines=3,
        max_lines=8,
    )

    # Ollama Enhancement - directly under prompt (the key UX change!)
    with gr.Group():
        with gr.Row():
            use_ollama = gr.Checkbox(
                label="Enhance with Ollama",
                value=False,
                scale=1,
                info="Use local LLM to expand your prompt"
            )
            models_list = get_ollama_models_list() if ollama_available else OLLAMA_MODELS
            # Ensure default model is in the list
            if DEFAULT_OLLAMA_MODEL not in models_list:
                models_list = [DEFAULT_OLLAMA_MODEL] + models_list
            ollama_model = gr.Dropdown(
                label="Model",
                choices=models_list,
                value=DEFAULT_OLLAMA_MODEL,
                scale=2,
                interactive=ollama_available,
                allow_custom_value=True
            )
        with gr.Row():
            ollama_length = gr.Dropdown(
                label="Length",
                choices=OLLAMA_LENGTH_OPTIONS,
                value="medium",
                scale=1,
                info="Output length"
            )
            ollama_complexity = gr.Dropdown(
                label="Complexity",
                choices=OLLAMA_COMPLEXITY_OPTIONS,
                value="detailed",
                scale=1,
                info="Detail level"
            )

    # Style Selection (part of "Prompt Modifiers" group)
    gr.Markdown("### Style & Modifiers")
    with gr.Row():
        style_presets = get_style_presets()
        style_dropdown = gr.Dropdown(
            label="Style Preset",
            choices=list(style_presets.keys()),
            value="None",
            scale=2,
            info="Appends style suffix to your prompt"
        )

    # Negative prompt
    negative_prompt = gr.Textbox(
        label="Negative Prompt (what to avoid)",
        placeholder="blurry, low quality, distorted...",
        lines=1,
    )

    # Settings loader (drag & drop JSON)
    gr.Markdown("### Load Settings")
    with gr.Row():
        settings_file = gr.File(
            label="Drop settings JSON here",
            file_types=[".json"],
            file_count="single",
            scale=2,
        )
        load_settings_btn = gr.Button(
            "Load Settings",
            variant="secondary",
            size="sm",
            scale=1,
        )

    return PromptInputComponents(
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        ollama_length=ollama_length,
        ollama_complexity=ollama_complexity,
        style_dropdown=style_dropdown,
        settings_file=settings_file,
        load_settings_btn=load_settings_btn,
    )


def enhance_prompt_with_ollama(
    prompt: str,
    model: str,
    length: str,
    complexity: str
) -> Tuple[str, str]:
    """Enhance a prompt using Ollama.

    Args:
        prompt: Original prompt
        model: Ollama model name
        length: Desired output length
        complexity: Desired complexity level

    Returns:
        Tuple of (enhanced_prompt, status_message)
    """
    state = get_state()

    if not state.ollama_available or not state.ollama_enhancer:
        return prompt, "Ollama not available"

    try:
        enhanced = state.ollama_enhancer.enhance(
            prompt,
            length=length,
            complexity=complexity
        )
        return enhanced, f"Enhanced ({length}/{complexity})"
    except Exception as e:
        return prompt, f"Enhancement failed: {e}"


def wire_prompt_input_events(
    components: PromptInputComponents,
    refresh_models_callback=None
) -> None:
    """Wire up event handlers for prompt input components.

    Args:
        components: PromptInputComponents instance to wire up
        refresh_models_callback: Optional callback to refresh model list
    """
    # Enable/disable Ollama controls based on checkbox
    def toggle_ollama_controls(use_ollama: bool):
        return [
            gr.update(interactive=use_ollama),
            gr.update(interactive=use_ollama),
            gr.update(interactive=use_ollama),
        ]

    components.use_ollama.change(
        fn=toggle_ollama_controls,
        inputs=[components.use_ollama],
        outputs=[
            components.ollama_model,
            components.ollama_length,
            components.ollama_complexity
        ]
    )
