"""
Generation Settings Component for HunyuanImage-3.0 UI.
Contains aspect ratio, quality, seed, and other generation parameters.
"""

import gradio as gr
from dataclasses import dataclass
from typing import List

from ui.constants import (
    ASPECT_RATIOS,
    QUALITY_PRESETS,
    DEFAULT_SEED,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_QUALITY,
)


@dataclass
class GenSettingsComponents:
    """Container for generation settings UI components."""
    aspect_ratio: gr.Dropdown
    quality: gr.Dropdown
    seed: gr.Number
    random_seed_btn: gr.Button
    batch_count: gr.Slider


def create_gen_settings() -> GenSettingsComponents:
    """Create the generation settings UI component.

    Returns:
        GenSettingsComponents with references to all widgets.
    """
    gr.Markdown("### Generation Settings")

    with gr.Row():
        aspect_ratio = gr.Dropdown(
            label="Aspect Ratio",
            choices=list(ASPECT_RATIOS.keys()),
            value=DEFAULT_ASPECT_RATIO,
            scale=1
        )
        quality = gr.Dropdown(
            label="Quality",
            choices=list(QUALITY_PRESETS.keys()),
            value=DEFAULT_QUALITY,
            scale=1,
            info="Higher = better but slower"
        )

    with gr.Row():
        seed = gr.Number(
            label="Seed (-1 = random)",
            value=DEFAULT_SEED,
            precision=0,
            scale=2
        )
        random_seed_btn = gr.Button(
            "Random",
            size="sm",
            scale=1
        )

    # Batch count for single generation mode
    batch_count = gr.Slider(
        label="Images to Generate",
        minimum=1,
        maximum=10,
        value=1,
        step=1,
        info="Generate multiple images with same settings"
    )

    return GenSettingsComponents(
        aspect_ratio=aspect_ratio,
        quality=quality,
        seed=seed,
        random_seed_btn=random_seed_btn,
        batch_count=batch_count,
    )


def get_steps_from_quality(quality_name: str) -> int:
    """Get number of steps from quality preset name."""
    if quality_name in QUALITY_PRESETS:
        return QUALITY_PRESETS[quality_name]["steps"]
    return 20  # Default


def get_size_from_aspect(aspect_name: str) -> str:
    """Get image size from aspect ratio name."""
    if aspect_name in ASPECT_RATIOS:
        return ASPECT_RATIOS[aspect_name]
    return "1024x1024"  # Default


def generate_random_seed() -> int:
    """Generate a random seed value."""
    import random
    return random.randint(0, 2**31 - 1)


def wire_gen_settings_events(components: GenSettingsComponents) -> None:
    """Wire up event handlers for generation settings.

    Args:
        components: GenSettingsComponents instance to wire up
    """
    # Random seed button
    components.random_seed_btn.click(
        fn=generate_random_seed,
        inputs=[],
        outputs=[components.seed]
    )
