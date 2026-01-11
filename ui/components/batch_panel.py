"""
Batch Panel Component for HunyuanImage-3.0 UI.
Contains all batch generation tabs: Configure, Prompt Runs, Combine, Browse, Examples, Wildcards.
Opens by default for batch-heavy workflows.
"""

import gradio as gr
from dataclasses import dataclass
from typing import List, Optional

from ui.constants import (
    ASPECT_RATIOS,
    QUALITY_PRESETS,
    DEFAULT_STYLE_PRESETS,
    OLLAMA_MODELS,
    OLLAMA_LENGTH_OPTIONS,
    OLLAMA_COMPLEXITY_OPTIONS,
)


@dataclass
class BatchConfigComponents:
    """Configure Batch tab components."""
    batch_name: gr.Textbox
    batch_themes: gr.Textbox
    batch_negative_prompt: gr.Textbox
    batch_variations: gr.Slider
    batch_images_per: gr.Slider
    batch_styles: gr.CheckboxGroup
    batch_aspect: gr.Dropdown
    batch_quality: gr.Dropdown
    batch_guidance_scale: gr.Slider
    batch_llm_backend: gr.Dropdown
    batch_enhance: gr.Checkbox
    batch_ollama_model: gr.Dropdown
    batch_ollama_length: gr.Dropdown
    batch_ollama_complexity: gr.Dropdown
    batch_random_seeds: gr.Checkbox
    # Controls
    batch_calculate_btn: gr.Button
    batch_start_btn: gr.Button
    batch_stop_btn: gr.Button
    generate_prompts_only_btn: gr.Button
    # Status
    batch_preview: gr.Markdown
    batch_status: gr.Textbox
    batch_output_dir: gr.Textbox
    batch_gallery: gr.Gallery
    # Config management
    config_name_input: gr.Textbox
    batch_save_btn: gr.Button
    batch_config_dropdown: gr.Dropdown
    batch_load_btn: gr.Button
    batch_config_status: gr.Textbox


@dataclass
class PromptRunComponents:
    """Saved Prompt Runs tab components."""
    prompt_run_dropdown: gr.Dropdown
    refresh_prompt_runs_btn: gr.Button
    interleave_checkbox: gr.Checkbox
    start_at_prompt: gr.Number
    loop_back_checkbox: gr.Checkbox
    run_from_saved_btn: gr.Button
    stop_prompt_run_btn: gr.Button
    delete_prompt_run_btn: gr.Button
    prompt_run_status: gr.Textbox
    prompt_run_output_dir: gr.Textbox
    prompt_run_preview: gr.Markdown
    prompt_run_gallery: gr.Gallery


@dataclass
class BrowseBatchComponents:
    """Browse Batches tab components."""
    batch_dir_dropdown: gr.Dropdown
    batch_dir_custom: gr.Textbox
    browse_refresh_btn: gr.Button
    browse_gallery: gr.Gallery
    browse_status: gr.Textbox


@dataclass
class BatchPanelComponents:
    """Container for all batch panel components."""
    config: BatchConfigComponents
    prompt_runs: PromptRunComponents
    browse: BrowseBatchComponents
    accordion: gr.Accordion


def get_style_choices() -> List[str]:
    """Get list of style preset names."""
    return list(DEFAULT_STYLE_PRESETS.keys())


def get_ollama_models() -> List[str]:
    """Get available Ollama models."""
    try:
        from ui.state import get_state
        state = get_state()
        if state.ollama_manager:
            models = state.ollama_manager.list_models()
            if models:
                return models
    except Exception:
        pass
    return OLLAMA_MODELS


def get_saved_configs() -> List[str]:
    """Get list of saved batch configs."""
    try:
        import batch_manager as bm
        return bm.get_config_choices()
    except Exception:
        return []


def get_saved_prompt_runs() -> List[str]:
    """Get list of saved prompt runs."""
    from ui.constants import OUTPUT_DIR
    prompt_runs_dir = OUTPUT_DIR / "prompt_runs"
    if not prompt_runs_dir.exists():
        return []
    runs = sorted(prompt_runs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [r.stem for r in runs]


def create_batch_config_tab() -> BatchConfigComponents:
    """Create the Configure Batch tab."""
    with gr.Row():
        # Left column - Configuration
        with gr.Column(scale=1):
            batch_name = gr.Textbox(
                label="Batch Name",
                value="my_batch",
                placeholder="Name for this batch run"
            )

            batch_themes = gr.Textbox(
                label="Themes/Prompts (one per line)",
                placeholder="cyberpunk city at night\nunderwater ancient temple\nfantasy forest",
                lines=5,
                max_lines=15
            )

            batch_negative_prompt = gr.Textbox(
                label="Negative Prompt (all images)",
                placeholder="blurry, low quality, distorted",
                lines=1
            )

            with gr.Row():
                batch_variations = gr.Slider(
                    label="Variations per theme",
                    minimum=1, maximum=100, value=3, step=1
                )
                batch_images_per = gr.Slider(
                    label="Images per combo",
                    minimum=1, maximum=50, value=1, step=1
                )

            batch_styles = gr.CheckboxGroup(
                label="Styles (select multiple)",
                choices=get_style_choices(),
                value=["Photorealistic", "Cinematic"]
            )

            with gr.Row():
                batch_aspect = gr.Dropdown(
                    label="Aspect", choices=list(ASPECT_RATIOS.keys()),
                    value="1:1 (Square)"
                )
                batch_quality = gr.Dropdown(
                    label="Quality", choices=list(QUALITY_PRESETS.keys()),
                    value="Standard"
                )
                batch_guidance_scale = gr.Slider(
                    label="Guidance", minimum=1.0, maximum=15.0,
                    value=5.0, step=0.5
                )

            with gr.Row():
                batch_llm_backend = gr.Dropdown(
                    label="LLM Backend",
                    choices=["Ollama", "LM Studio"],
                    value="Ollama"
                )
                batch_enhance = gr.Checkbox(
                    label="Enhance prompts", value=True
                )

            with gr.Row():
                batch_ollama_model = gr.Dropdown(
                    label="Model",
                    choices=get_ollama_models(),
                    value="qwen2.5:7b-instruct",
                    allow_custom_value=True
                )

            with gr.Row():
                batch_ollama_length = gr.Dropdown(
                    label="Length",
                    choices=OLLAMA_LENGTH_OPTIONS,
                    value="medium"
                )
                batch_ollama_complexity = gr.Dropdown(
                    label="Complexity",
                    choices=OLLAMA_COMPLEXITY_OPTIONS,
                    value="detailed"
                )

            batch_random_seeds = gr.Checkbox(
                label="Random seeds", value=True
            )

        # Right column - Preview and controls
        with gr.Column(scale=1):
            batch_preview = gr.Markdown("Enter themes to see preview...")

            with gr.Row():
                batch_calculate_btn = gr.Button("Calculate", size="sm")
                generate_prompts_only_btn = gr.Button(
                    "Generate Prompts Only", variant="secondary"
                )

            with gr.Row():
                batch_start_btn = gr.Button("Start Batch", variant="primary", size="lg")
                batch_stop_btn = gr.Button("Stop", variant="stop")

            batch_status = gr.Textbox(
                label="Status", value="Ready", interactive=False, lines=2
            )
            batch_output_dir = gr.Textbox(
                label="Output Directory", interactive=False
            )

            gr.Markdown("---\n**Config Management**")
            with gr.Row():
                config_name_input = gr.Textbox(
                    label="Config Name", placeholder="my_config", scale=2
                )
                batch_save_btn = gr.Button("Save", variant="primary", size="sm")

            with gr.Row():
                batch_config_dropdown = gr.Dropdown(
                    label="Saved Configs",
                    choices=get_saved_configs(),
                    scale=2
                )
                batch_load_btn = gr.Button("Load", size="sm")

            batch_config_status = gr.Textbox(
                label="Config Status", interactive=False
            )

    gr.Markdown("---\n**Batch Progress**")
    batch_gallery = gr.Gallery(
        label="Generated Images", columns=6, rows=2, height=180
    )

    return BatchConfigComponents(
        batch_name=batch_name,
        batch_themes=batch_themes,
        batch_negative_prompt=batch_negative_prompt,
        batch_variations=batch_variations,
        batch_images_per=batch_images_per,
        batch_styles=batch_styles,
        batch_aspect=batch_aspect,
        batch_quality=batch_quality,
        batch_guidance_scale=batch_guidance_scale,
        batch_llm_backend=batch_llm_backend,
        batch_enhance=batch_enhance,
        batch_ollama_model=batch_ollama_model,
        batch_ollama_length=batch_ollama_length,
        batch_ollama_complexity=batch_ollama_complexity,
        batch_random_seeds=batch_random_seeds,
        batch_calculate_btn=batch_calculate_btn,
        batch_start_btn=batch_start_btn,
        batch_stop_btn=batch_stop_btn,
        generate_prompts_only_btn=generate_prompts_only_btn,
        batch_preview=batch_preview,
        batch_status=batch_status,
        batch_output_dir=batch_output_dir,
        batch_gallery=batch_gallery,
        config_name_input=config_name_input,
        batch_save_btn=batch_save_btn,
        batch_config_dropdown=batch_config_dropdown,
        batch_load_btn=batch_load_btn,
        batch_config_status=batch_config_status,
    )


def create_prompt_runs_tab() -> PromptRunComponents:
    """Create the Saved Prompt Runs tab."""
    gr.Markdown("""
    **Load and run saved prompt lists**
    Seeds are generated fresh each run for variety.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                prompt_run_dropdown = gr.Dropdown(
                    label="Select Prompt Run",
                    choices=get_saved_prompt_runs(),
                    scale=3
                )
                refresh_prompt_runs_btn = gr.Button("↻", size="sm")

            with gr.Row():
                interleave_checkbox = gr.Checkbox(
                    label="Interleave prompts", value=True,
                    info="Alternate themes instead of sequential"
                )

            with gr.Row():
                start_at_prompt = gr.Number(
                    label="Start at #", value=1, minimum=1
                )
                loop_back_checkbox = gr.Checkbox(
                    label="Loop", value=False,
                    info="Continue indefinitely"
                )

            with gr.Row():
                run_from_saved_btn = gr.Button(
                    "Run Selected", variant="primary", size="lg"
                )
                stop_prompt_run_btn = gr.Button("Stop", variant="stop")

            delete_prompt_run_btn = gr.Button("Delete Run", variant="stop", size="sm")

            prompt_run_status = gr.Textbox(
                label="Status", interactive=False, lines=2
            )
            prompt_run_output_dir = gr.Textbox(
                label="Output Directory", interactive=False
            )

        with gr.Column(scale=1):
            prompt_run_preview = gr.Markdown("Select a run to preview...")

    gr.Markdown("---\n**Progress**")
    prompt_run_gallery = gr.Gallery(columns=6, rows=2, height=180)

    return PromptRunComponents(
        prompt_run_dropdown=prompt_run_dropdown,
        refresh_prompt_runs_btn=refresh_prompt_runs_btn,
        interleave_checkbox=interleave_checkbox,
        start_at_prompt=start_at_prompt,
        loop_back_checkbox=loop_back_checkbox,
        run_from_saved_btn=run_from_saved_btn,
        stop_prompt_run_btn=stop_prompt_run_btn,
        delete_prompt_run_btn=delete_prompt_run_btn,
        prompt_run_status=prompt_run_status,
        prompt_run_output_dir=prompt_run_output_dir,
        prompt_run_preview=prompt_run_preview,
        prompt_run_gallery=prompt_run_gallery,
    )


def create_browse_tab() -> BrowseBatchComponents:
    """Create the Browse Batches tab."""
    gr.Markdown("**Browse generated batches**")

    with gr.Row():
        batch_dir_dropdown = gr.Dropdown(
            label="Batch Directory",
            choices=[],
            scale=2
        )
        batch_dir_custom = gr.Textbox(
            label="Custom Path",
            placeholder="/path/to/batch",
            scale=2
        )
        browse_refresh_btn = gr.Button("↻", size="sm")

    browse_gallery = gr.Gallery(columns=6, rows=3, height=250)

    browse_status = gr.Textbox(
        label="Status", interactive=False
    )

    return BrowseBatchComponents(
        batch_dir_dropdown=batch_dir_dropdown,
        batch_dir_custom=batch_dir_custom,
        browse_refresh_btn=browse_refresh_btn,
        browse_gallery=browse_gallery,
        browse_status=browse_status,
    )


def create_batch_panel(open_by_default: bool = True) -> BatchPanelComponents:
    """Create the complete batch panel with all tabs.

    Args:
        open_by_default: Whether accordion starts open (True for batch-heavy workflows)

    Returns:
        BatchPanelComponents with references to all widgets.
    """
    with gr.Accordion("Batch Generation", open=open_by_default) as accordion:
        gr.Markdown("""
        **Mass image generation with themes, styles, and LLM enhancement**
        """)

        with gr.Tabs():
            with gr.Tab("Configure"):
                config = create_batch_config_tab()

            with gr.Tab("Prompt Runs"):
                prompt_runs = create_prompt_runs_tab()

            with gr.Tab("Browse"):
                browse = create_browse_tab()

            with gr.Tab("Examples"):
                gr.Markdown("""
                **Example themes to try:**

                ```
                cyberpunk city at night with neon signs
                underwater ancient temple with bioluminescent creatures
                fantasy forest with magical creatures and floating lights
                steampunk airship flying over Victorian London
                alien landscape with multiple moons and strange vegetation
                ```

                **Tips:**
                - Use specific, descriptive themes
                - Let Ollama add the creative details
                - Mix different style presets for variety
                """)

    return BatchPanelComponents(
        config=config,
        prompt_runs=prompt_runs,
        browse=browse,
        accordion=accordion,
    )
