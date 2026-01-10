#!/usr/bin/env python3
"""
Batch Manager Module for HunyuanImage-3.0 UI
Handles batch configuration management, image editing, and batch operations.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import gradio as gr

# Output directory (shared with main UI)
OUTPUT_DIR = Path("/media/james/DataDrive/jamesw767/Hun3d/outputs")
BATCH_CONFIGS_DIR = OUTPUT_DIR / "batch_configs"
BATCHES_DIR = OUTPUT_DIR / "batches"

# Ensure directories exist
BATCH_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
BATCHES_DIR.mkdir(parents=True, exist_ok=True)

# Global state for selected images
selected_images: List[str] = []
current_batch_path: Optional[Path] = None


# ============================================================================
# CONFIG MANAGEMENT FUNCTIONS
# ============================================================================

def get_saved_batch_configs_with_info() -> List[Dict]:
    """Get list of saved batch configs with metadata for display"""
    if not BATCH_CONFIGS_DIR.exists():
        return []

    configs = []
    for f in sorted(BATCH_CONFIGS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)

            themes_count = len(data.get('themes', '').split('\n')) if data.get('themes') else 0
            saved_at = data.get('saved_at', '')[:10] if data.get('saved_at') else 'unknown'

            configs.append({
                'filename': f.name,
                'batch_name': data.get('batch_name', f.stem),
                'themes_count': themes_count,
                'styles': data.get('styles', []),
                'saved_at': saved_at,
                'display_name': f"{data.get('batch_name', f.stem)} ({themes_count} themes, {saved_at})"
            })
        except Exception:
            configs.append({
                'filename': f.name,
                'batch_name': f.stem,
                'themes_count': 0,
                'styles': [],
                'saved_at': 'error',
                'display_name': f"{f.stem} (error reading)"
            })

    return configs


def get_config_choices() -> List[str]:
    """Get list of config filenames for dropdown"""
    configs = get_saved_batch_configs_with_info()
    return [c['filename'] for c in configs]


def get_config_display_choices() -> List[Tuple[str, str]]:
    """Get list of (display_name, filename) tuples for dropdown"""
    configs = get_saved_batch_configs_with_info()
    return [(c['display_name'], c['filename']) for c in configs]


def save_batch_config_named(
    config_name: str,
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
    negative_prompt: str = "",
    ollama_length: str = "medium",
    ollama_complexity: str = "detailed"
) -> Tuple[str, Any]:
    """Save batch configuration with a specific name"""
    if not config_name.strip():
        config_name = batch_name if batch_name.strip() else "untitled"

    config = {
        "batch_name": batch_name,
        "themes": themes_text,
        "variations_per_theme": variations_per_theme,
        "styles": styles_selected,
        "images_per_combo": images_per_combo,
        "llm_backend": llm_backend,
        "ollama_model": ollama_model,
        "enhance_prompts": enhance_prompts,
        "aspect_ratio": aspect_ratio,
        "quality_preset": quality_preset,
        "random_seeds": random_seeds,
        "negative_prompt": negative_prompt,
        "ollama_length": ollama_length,
        "ollama_complexity": ollama_complexity,
        "saved_at": datetime.now().isoformat()
    }

    # Sanitize filename
    safe_name = "".join(c for c in config_name[:40] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    if not safe_name:
        safe_name = "config"
    filename = f"{safe_name}.json"
    filepath = BATCH_CONFIGS_DIR / filename

    # Check if overwriting
    is_overwrite = filepath.exists()

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    action = "Overwritten" if is_overwrite else "Saved"
    return f"{action}: {filename}", gr.update(choices=get_config_choices(), value=filename)


def delete_batch_config(config_name: str) -> Tuple[str, Any]:
    """Delete a saved batch configuration"""
    if not config_name:
        return "No config selected", gr.update()

    filepath = BATCH_CONFIGS_DIR / config_name
    if not filepath.exists():
        return f"Config not found: {config_name}", gr.update()

    try:
        filepath.unlink()
        choices = get_config_choices()
        return f"Deleted: {config_name}", gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        return f"Error deleting: {e}", gr.update()


def rename_batch_config(old_name: str, new_name: str) -> Tuple[str, Any]:
    """Rename a batch configuration file"""
    if not old_name:
        return "No config selected", gr.update()
    if not new_name.strip():
        return "Please enter a new name", gr.update()

    old_path = BATCH_CONFIGS_DIR / old_name
    if not old_path.exists():
        return f"Config not found: {old_name}", gr.update()

    # Sanitize new name
    safe_name = "".join(c for c in new_name[:40] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    if not safe_name:
        return "Invalid name", gr.update()

    new_filename = f"{safe_name}.json"
    new_path = BATCH_CONFIGS_DIR / new_filename

    if new_path.exists() and new_path != old_path:
        return f"A config named '{new_filename}' already exists", gr.update()

    try:
        old_path.rename(new_path)
        choices = get_config_choices()
        return f"Renamed: {old_name} -> {new_filename}", gr.update(choices=choices, value=new_filename)
    except Exception as e:
        return f"Error renaming: {e}", gr.update()


def load_batch_config_full(config_name: str) -> List[Any]:
    """Load a saved batch configuration (extended version with all fields)"""
    if not config_name:
        return [gr.update()] * 15 + ["No config selected"]

    filepath = BATCH_CONFIGS_DIR / config_name
    if not filepath.exists():
        return [gr.update()] * 15 + [f"Config not found: {config_name}"]

    try:
        with open(filepath, 'r') as f:
            config = json.load(f)

        themes_count = len(config.get('themes', '').split('\n')) if config.get('themes') else 0

        return [
            gr.update(value=config.get('batch_name', '')),
            gr.update(value=config.get('themes', '')),
            gr.update(value=config.get('variations_per_theme', 3)),
            gr.update(value=config.get('styles', [])),
            gr.update(value=config.get('images_per_combo', 1)),
            gr.update(value=config.get('llm_backend', 'Ollama')),
            gr.update(value=config.get('ollama_model', 'qwen2.5:7b-instruct')),
            gr.update(value=config.get('enhance_prompts', True)),
            gr.update(value=config.get('aspect_ratio', '1:1 (Square)')),
            gr.update(value=config.get('quality_preset', 'Standard')),
            gr.update(value=config.get('random_seeds', True)),
            gr.update(value=config.get('negative_prompt', '')),
            gr.update(value=config.get('ollama_length', 'medium')),
            gr.update(value=config.get('ollama_complexity', 'detailed')),
            gr.update(value=config_name.replace('.json', '')),  # Set config name field
            f"Loaded: {config_name}\nThemes: {themes_count} | Saved: {config.get('saved_at', 'unknown')[:19]}"
        ]
    except Exception as e:
        return [gr.update()] * 15 + [f"Error loading: {e}"]


# ============================================================================
# BATCH DIRECTORY MANAGEMENT
# ============================================================================

def get_batch_directories_with_info(custom_dir: str = None) -> List[Dict]:
    """Get list of batch directories with metadata"""
    base_dir = Path(custom_dir) if custom_dir else BATCHES_DIR

    if not base_dir.exists():
        return []

    batches = []

    # Check if this directory itself is a batch
    if list(base_dir.glob("*.png")):
        batches.append({
            'name': base_dir.name,
            'path': str(base_dir),
            'image_count': len(list(base_dir.glob("*.png"))),
            'has_manifest': (base_dir / "manifest.json").exists()
        })
        return batches

    # Look for subdirectories
    for d in sorted(base_dir.iterdir(), key=os.path.getmtime, reverse=True):
        if d.is_dir():
            images = list(d.glob("*.png"))
            if images:
                batches.append({
                    'name': d.name,
                    'path': str(d),
                    'image_count': len(images),
                    'has_manifest': (d / "manifest.json").exists()
                })

    return batches


def delete_batch_directory(batch_name: str, custom_base: str = None) -> Tuple[str, Any]:
    """Delete an entire batch directory"""
    if not batch_name or batch_name == "(No batches yet)":
        return "No batch selected", gr.update()

    base_dir = Path(custom_base) if custom_base else BATCHES_DIR
    batch_path = base_dir / batch_name

    if not batch_path.exists():
        return f"Batch not found: {batch_name}", gr.update()

    # Count items to be deleted
    image_count = len(list(batch_path.glob("*.png")))

    try:
        shutil.rmtree(batch_path)
        batches = get_batch_directories_with_info(custom_base)
        choices = [b['name'] for b in batches] if batches else ["(No batches yet)"]
        return f"Deleted batch: {batch_name} ({image_count} images)", gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        return f"Error deleting batch: {e}", gr.update()


def load_batch_manifest(batch_name: str, custom_base: str = None) -> Dict:
    """Load the manifest.json from a batch directory"""
    base_dir = Path(custom_base) if custom_base else BATCHES_DIR
    manifest_path = base_dir / batch_name / "manifest.json"

    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def load_batch_settings_to_ui(batch_name: str, custom_base: str = None) -> List[Any]:
    """Load batch manifest settings into the UI for re-running"""
    manifest = load_batch_manifest(batch_name, custom_base)

    if not manifest:
        return [gr.update()] * 14 + ["No manifest found for this batch"]

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
        gr.update(value=True),  # random_seeds default
        gr.update(value=settings.get('negative_prompt', '')),
        gr.update(value=settings.get('ollama_length', 'medium')),
        gr.update(value=settings.get('ollama_complexity', 'detailed')),
        f"Loaded settings from: {batch_name}\nOriginal: {manifest.get('total_images', '?')} images"
    ]


# ============================================================================
# INDIVIDUAL IMAGE MANAGEMENT
# ============================================================================

def get_image_config(image_path: str) -> Dict:
    """Load the JSON config for a specific image"""
    config_path = Path(image_path).with_suffix('.json')

    if not config_path.exists():
        # Try to extract info from filename
        return {'error': 'No config file found', 'image_path': image_path}

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {'error': str(e), 'image_path': image_path}


def save_image_config(image_path: str, config: Dict) -> str:
    """Save updated config for an image"""
    config_path = Path(image_path).with_suffix('.json')
    config['image_path'] = str(image_path)
    config['modified_at'] = datetime.now().isoformat()

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return f"Config saved for: {Path(image_path).name}"
    except Exception as e:
        return f"Error saving config: {e}"


def delete_image(image_path: str) -> Tuple[str, bool]:
    """Delete an image and its config file"""
    image_path = Path(image_path)
    config_path = image_path.with_suffix('.json')

    deleted = []
    errors = []

    if image_path.exists():
        try:
            image_path.unlink()
            deleted.append(image_path.name)
        except Exception as e:
            errors.append(f"Image: {e}")

    if config_path.exists():
        try:
            config_path.unlink()
            deleted.append(config_path.name)
        except Exception as e:
            errors.append(f"Config: {e}")

    if errors:
        return f"Errors: {', '.join(errors)}", False
    if deleted:
        return f"Deleted: {', '.join(deleted)}", True
    return "Files not found", False


def delete_multiple_images(image_paths: List[str]) -> str:
    """Delete multiple images and their configs"""
    success_count = 0
    error_count = 0

    for path in image_paths:
        _, ok = delete_image(path)
        if ok:
            success_count += 1
        else:
            error_count += 1

    return f"Deleted {success_count} images, {error_count} errors"


def get_image_info_display(image_path: str) -> str:
    """Get formatted display of image config"""
    config = get_image_config(image_path)

    if 'error' in config:
        return f"Error: {config['error']}\nImage: {image_path}"

    lines = [
        f"**Image:** {Path(image_path).name}",
        f"**Prompt:** {config.get('prompt', 'N/A')[:100]}...",
        f"**Style:** {config.get('style', 'None')}",
        f"**Seed:** {config.get('seed', 'N/A')}",
        f"**Size:** {config.get('image_size', 'N/A')}",
        f"**Steps:** {config.get('steps', 'N/A')}",
        f"**Quality:** {config.get('quality_preset', 'N/A')}",
    ]

    if config.get('generation_time'):
        lines.append(f"**Gen Time:** {config['generation_time']:.1f}s")

    if config.get('wildcards_used'):
        lines.append(f"**Wildcards:** Yes")
        if config.get('processed_prompt'):
            lines.append(f"**Processed:** {config['processed_prompt'][:80]}...")

    return "\n".join(lines)


def get_image_config_json(image_path: str) -> str:
    """Get the raw JSON config for editing"""
    config = get_image_config(image_path)
    return json.dumps(config, indent=2)


def update_image_config_from_json(image_path: str, json_str: str) -> str:
    """Update image config from edited JSON"""
    try:
        config = json.loads(json_str)
        return save_image_config(image_path, config)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        return f"Error saving: {e}"


# ============================================================================
# BATCH IMAGE BROWSER WITH SELECTION
# ============================================================================

def get_batch_images_with_selection(
    batch_name: str,
    page: int = 0,
    per_page: int = 24,
    custom_base: str = None
) -> Tuple[List[str], str, int, int, List[Dict]]:
    """Get images from a batch with metadata for selection UI"""
    if not batch_name or batch_name == "(No batches yet)":
        return [], "No batch selected", 0, 0, []

    base_dir = Path(custom_base) if custom_base else BATCHES_DIR
    batch_path = base_dir / batch_name

    if not batch_path.exists():
        return [], f"Batch not found: {batch_name}", 0, 0, []

    all_images = sorted(batch_path.glob("*.png"))
    total_images = len(all_images)
    total_pages = max(1, (total_images + per_page - 1) // per_page)

    page = max(0, min(page, total_pages - 1))
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, total_images)

    page_images = [str(img) for img in all_images[start_idx:end_idx]]

    # Get basic info for each image
    image_info = []
    for img_path in page_images:
        config = get_image_config(img_path)
        image_info.append({
            'path': img_path,
            'name': Path(img_path).name,
            'seed': config.get('seed', 'N/A'),
            'style': config.get('style', 'N/A'),
            'prompt_preview': config.get('prompt', '')[:40] + '...' if config.get('prompt') else 'N/A'
        })

    info = f"**{batch_name}**\nImages {start_idx + 1}-{end_idx} of {total_images} | Page {page + 1}/{total_pages}"

    return page_images, info, page, total_pages, image_info


# ============================================================================
# RERUN FUNCTIONALITY
# ============================================================================

def get_failed_images(batch_name: str, custom_base: str = None) -> List[Dict]:
    """Get list of failed images from results.json"""
    base_dir = Path(custom_base) if custom_base else BATCHES_DIR
    results_path = base_dir / batch_name / "results.json"

    if not results_path.exists():
        return []

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        return [r for r in results if 'error' in r]
    except Exception:
        return []


def prepare_rerun_prompts(batch_name: str, custom_base: str = None, failed_only: bool = False) -> List[Dict]:
    """Prepare prompts for re-running a batch"""
    manifest = load_batch_manifest(batch_name, custom_base)

    if not manifest:
        return []

    if failed_only:
        failed = get_failed_images(batch_name, custom_base)
        # Return the failed prompts
        return [{'prompt': f.get('prompt', ''), 'style': f.get('style', 'None')} for f in failed]

    # Return all prompts from manifest
    return manifest.get('prompts', [])


# ============================================================================
# UI COMPONENT BUILDERS
# ============================================================================

def create_config_manager_ui():
    """Create the config management UI components (to be called from main UI)"""
    # This function returns the components needed for config management
    # The actual Gradio components will be created in the main UI file
    pass


def create_batch_browser_ui():
    """Create the batch browser UI components (to be called from main UI)"""
    # This function returns the components needed for batch browsing
    # The actual Gradio components will be created in the main UI file
    pass
