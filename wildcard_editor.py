#!/usr/bin/env python3
"""
Wildcard Editor Module for HunyuanImage-3.0 UI
Handles wildcard CRUD operations and LLM-based generation.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import gradio as gr

# Paths
WILDCARDS_FILE = Path(__file__).parent / "wildcards.json"
WILDCARDS_BACKUP_DIR = Path(__file__).parent / "wildcards_backups"
WILDCARDS_BACKUP_DIR.mkdir(exist_ok=True)


def load_wildcards() -> Dict[str, List[str]]:
    """Load wildcards from JSON file"""
    if not WILDCARDS_FILE.exists():
        return {}
    try:
        with open(WILDCARDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading wildcards: {e}")
        return {}


def save_wildcards(data: Dict[str, List[str]], create_backup: bool = True) -> str:
    """Save wildcards to JSON file with optional backup"""
    try:
        # Create backup before saving
        if create_backup and WILDCARDS_FILE.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = WILDCARDS_BACKUP_DIR / f"wildcards_{timestamp}.json"
            with open(WILDCARDS_FILE, 'r', encoding='utf-8') as f:
                backup_data = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(backup_data)

            # Keep only last 10 backups
            backups = sorted(WILDCARDS_BACKUP_DIR.glob("*.json"), reverse=True)
            for old_backup in backups[10:]:
                old_backup.unlink()

        # Save the new data
        with open(WILDCARDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return "Saved successfully"
    except Exception as e:
        return f"Error saving: {e}"


def get_category_list() -> List[str]:
    """Get sorted list of category names"""
    data = load_wildcards()
    return sorted(data.keys())


def get_category_info(category: str) -> Tuple[int, List[str]]:
    """Get item count and preview for a category"""
    data = load_wildcards()
    items = data.get(category, [])
    return len(items), items[:20]


def get_category_items(category: str) -> List[str]:
    """Get all items in a category"""
    data = load_wildcards()
    return data.get(category, [])


def get_category_display_list() -> List[Tuple[str, str]]:
    """Get list of (display_name, category_name) for dropdown"""
    data = load_wildcards()
    result = []
    for cat in sorted(data.keys()):
        count = len(data[cat])
        result.append((f"{cat} ({count} items)", cat))
    return result


# ============================================================================
# CATEGORY OPERATIONS
# ============================================================================

def add_category(category_name: str, initial_items: str = "") -> Tuple[str, Any]:
    """Add a new wildcard category"""
    if not category_name or not category_name.strip():
        return "Please enter a category name", gr.update()

    # Sanitize category name
    category_name = category_name.strip().lower()
    category_name = re.sub(r'[^a-z0-9\-]', '-', category_name)
    category_name = re.sub(r'-+', '-', category_name).strip('-')

    if not category_name:
        return "Invalid category name", gr.update()

    data = load_wildcards()

    if category_name in data:
        return f"Category '{category_name}' already exists", gr.update()

    # Parse initial items
    items = []
    if initial_items and initial_items.strip():
        items = [item.strip() for item in initial_items.split('\n') if item.strip()]

    data[category_name] = items
    msg = save_wildcards(data)

    if "Error" in msg:
        return msg, gr.update()

    categories = get_category_list()
    return f"Created category '{category_name}' with {len(items)} items", gr.update(choices=categories, value=category_name)


def delete_category(category_name: str) -> Tuple[str, Any, str]:
    """Delete an entire category"""
    if not category_name:
        return "No category selected", gr.update(), ""

    data = load_wildcards()

    if category_name not in data:
        return f"Category '{category_name}' not found", gr.update(), ""

    item_count = len(data[category_name])
    del data[category_name]
    msg = save_wildcards(data)

    if "Error" in msg:
        return msg, gr.update(), ""

    categories = get_category_list()
    return f"Deleted category '{category_name}' ({item_count} items)", gr.update(choices=categories, value=categories[0] if categories else None), ""


def rename_category(old_name: str, new_name: str) -> Tuple[str, Any]:
    """Rename a category"""
    if not old_name:
        return "No category selected", gr.update()
    if not new_name or not new_name.strip():
        return "Please enter a new name", gr.update()

    # Sanitize new name
    new_name = new_name.strip().lower()
    new_name = re.sub(r'[^a-z0-9\-]', '-', new_name)
    new_name = re.sub(r'-+', '-', new_name).strip('-')

    if not new_name:
        return "Invalid new name", gr.update()

    data = load_wildcards()

    if old_name not in data:
        return f"Category '{old_name}' not found", gr.update()

    if new_name in data and new_name != old_name:
        return f"Category '{new_name}' already exists", gr.update()

    # Rename by copying and deleting
    data[new_name] = data[old_name]
    if new_name != old_name:
        del data[old_name]

    msg = save_wildcards(data)

    if "Error" in msg:
        return msg, gr.update()

    categories = get_category_list()
    return f"Renamed '{old_name}' to '{new_name}'", gr.update(choices=categories, value=new_name)


# ============================================================================
# ITEM OPERATIONS
# ============================================================================

def add_items_to_category(category: str, items_text: str) -> Tuple[str, str]:
    """Add items to an existing category"""
    if not category:
        return "No category selected", ""
    if not items_text or not items_text.strip():
        return "No items to add", ""

    data = load_wildcards()

    if category not in data:
        return f"Category '{category}' not found", ""

    # Parse items
    new_items = [item.strip() for item in items_text.split('\n') if item.strip()]

    if not new_items:
        return "No valid items to add", ""

    # Add only unique items
    existing = set(data[category])
    added = []
    for item in new_items:
        if item not in existing:
            data[category].append(item)
            existing.add(item)
            added.append(item)

    if not added:
        return "All items already exist in category", ""

    # Sort the category
    data[category] = sorted(data[category])

    msg = save_wildcards(data)

    if "Error" in msg:
        return msg, ""

    return f"Added {len(added)} items to '{category}'", "\n".join(sorted(data[category]))


def remove_items_from_category(category: str, items_text: str) -> Tuple[str, str]:
    """Remove specific items from a category"""
    if not category:
        return "No category selected", ""
    if not items_text or not items_text.strip():
        return "No items to remove", ""

    data = load_wildcards()

    if category not in data:
        return f"Category '{category}' not found", ""

    # Parse items to remove
    items_to_remove = set(item.strip().lower() for item in items_text.split('\n') if item.strip())

    # Remove matching items (case-insensitive)
    original_count = len(data[category])
    data[category] = [item for item in data[category] if item.lower() not in items_to_remove]
    removed_count = original_count - len(data[category])

    if removed_count == 0:
        return "No matching items found to remove", "\n".join(data[category])

    msg = save_wildcards(data)

    if "Error" in msg:
        return msg, ""

    return f"Removed {removed_count} items from '{category}'", "\n".join(sorted(data[category]))


def replace_category_items(category: str, items_text: str) -> Tuple[str, str]:
    """Replace all items in a category"""
    if not category:
        return "No category selected", ""

    data = load_wildcards()

    if category not in data:
        return f"Category '{category}' not found", ""

    # Parse new items
    new_items = [item.strip() for item in items_text.split('\n') if item.strip()]

    data[category] = sorted(list(set(new_items)))  # Unique and sorted

    msg = save_wildcards(data)

    if "Error" in msg:
        return msg, ""

    return f"Replaced items in '{category}' ({len(data[category])} items)", "\n".join(data[category])


def search_items_across_categories(query: str) -> str:
    """Search for items across all categories"""
    if not query or not query.strip():
        return "Enter a search query"

    query = query.lower().strip()
    data = load_wildcards()

    results = []
    for category, items in data.items():
        matches = [item for item in items if query in item.lower()]
        if matches:
            results.append(f"**{category}** ({len(matches)} matches):")
            for match in matches[:10]:  # Limit to 10 per category
                results.append(f"  - {match}")
            if len(matches) > 10:
                results.append(f"  ... and {len(matches) - 10} more")
            results.append("")

    if not results:
        return f"No items found matching '{query}'"

    return "\n".join(results)


# ============================================================================
# LLM-BASED GENERATION
# ============================================================================

def generate_wildcards_with_llm(
    prompt: str,
    category_name: str,
    ollama_model: str = "qwen2.5:7b-instruct",
    count: int = 50,
    add_to_existing: bool = True
) -> Tuple[str, str, Any]:
    """Use Ollama to generate wildcard items based on a prompt"""
    if not prompt or not prompt.strip():
        return "Please enter a prompt describing what wildcards to generate", "", gr.update()

    if not category_name or not category_name.strip():
        return "Please enter a category name", "", gr.update()

    # Sanitize category name
    category_name = category_name.strip().lower()
    category_name = re.sub(r'[^a-z0-9\-]', '-', category_name)
    category_name = re.sub(r'-+', '-', category_name).strip('-')

    try:
        import requests

        # Build the system prompt for generating wildcards
        system_prompt = f"""You are a helpful assistant that generates lists of items for image generation prompts.
The user will ask for a list of items (like "types of metals" or "fantasy creatures").
You must respond with ONLY a simple list, one item per line.
Do not include numbers, bullets, or explanations.
Generate exactly {count} unique items.
Keep items concise (1-4 words each).
Do not include duplicates.
"""

        user_prompt = f"List {count} {prompt}. One item per line, no numbers or bullets."

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": ollama_model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 2000
                }
            },
            timeout=120
        )

        if response.status_code != 200:
            return f"Ollama error: {response.status_code}", "", gr.update()

        result = response.json()
        generated_text = result.get('response', '')

        # Parse the generated items
        lines = generated_text.strip().split('\n')
        items = []
        for line in lines:
            # Clean up the line - remove numbers, bullets, etc.
            item = line.strip()
            item = re.sub(r'^[\d]+[.\)]\s*', '', item)  # Remove "1. " or "1) "
            item = re.sub(r'^[-*•]\s*', '', item)  # Remove "- " or "* "
            item = item.strip().lower()
            if item and len(item) > 1 and len(item) < 100:
                items.append(item)

        # Remove duplicates
        items = list(dict.fromkeys(items))

        if not items:
            return "No valid items generated. Try a different prompt.", "", gr.update()

        # Add to wildcards
        data = load_wildcards()

        if add_to_existing and category_name in data:
            # Add to existing category
            existing = set(data[category_name])
            new_items = [item for item in items if item not in existing]
            data[category_name].extend(new_items)
            data[category_name] = sorted(data[category_name])
            action = f"Added {len(new_items)} new items to existing"
        else:
            # Create new or replace
            data[category_name] = sorted(items)
            action = "Created new" if category_name not in data else "Replaced"

        msg = save_wildcards(data)

        if "Error" in msg:
            return msg, "\n".join(items), gr.update()

        categories = get_category_list()
        return f"{action} category '{category_name}' with {len(items)} items", "\n".join(items), gr.update(choices=categories, value=category_name)

    except requests.exceptions.ConnectionError:
        return "Ollama not running. Start Ollama first.", "", gr.update()
    except requests.exceptions.Timeout:
        return "Ollama request timed out. Try again.", "", gr.update()
    except Exception as e:
        return f"Error: {e}", "", gr.update()


def suggest_category_name(prompt: str) -> str:
    """Suggest a category name based on the prompt"""
    if not prompt:
        return ""

    # Simple heuristic: extract key words
    prompt_lower = prompt.lower()

    # Remove common words
    stop_words = {'all', 'the', 'a', 'an', 'of', 'types', 'kinds', 'list', 'give', 'me',
                  'different', 'various', 'some', 'many', 'few', 'examples'}

    words = re.findall(r'[a-z]+', prompt_lower)
    key_words = [w for w in words if w not in stop_words and len(w) > 2]

    if key_words:
        # Use first 1-2 meaningful words
        name = '-'.join(key_words[:2])
        return name

    return ""


# ============================================================================
# BACKUP MANAGEMENT
# ============================================================================

def get_backup_list() -> List[str]:
    """Get list of available backups"""
    if not WILDCARDS_BACKUP_DIR.exists():
        return []
    backups = sorted(WILDCARDS_BACKUP_DIR.glob("*.json"), reverse=True)
    return [b.name for b in backups]


def restore_from_backup(backup_name: str) -> Tuple[str, Any]:
    """Restore wildcards from a backup file"""
    if not backup_name:
        return "No backup selected", gr.update()

    backup_path = WILDCARDS_BACKUP_DIR / backup_name
    if not backup_path.exists():
        return f"Backup not found: {backup_name}", gr.update()

    try:
        # Load the backup
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)

        # Save current as backup first
        save_wildcards(load_wildcards(), create_backup=True)

        # Restore from backup (without creating another backup)
        with open(WILDCARDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        categories = get_category_list()
        return f"Restored from {backup_name} ({len(backup_data)} categories)", gr.update(choices=categories)

    except Exception as e:
        return f"Error restoring: {e}", gr.update()


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def load_category_for_editing(category: str) -> Tuple[str, str, str]:
    """Load a category's items for editing"""
    if not category:
        return "", "", "Select a category"

    data = load_wildcards()
    items = data.get(category, [])

    info = f"**{category}**: {len(items)} items"
    items_text = "\n".join(sorted(items))

    return category, items_text, info


def get_stats() -> str:
    """Get wildcard statistics"""
    data = load_wildcards()
    total_items = sum(len(items) for items in data.values())

    # Find largest/smallest categories
    if data:
        sorted_cats = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)
        largest = sorted_cats[:5]
        smallest = sorted_cats[-5:]
    else:
        largest = smallest = []

    stats = [
        f"**Total Categories:** {len(data)}",
        f"**Total Items:** {total_items:,}",
        f"**Average Items/Category:** {total_items // len(data) if data else 0}",
        "",
        "**Largest Categories:**"
    ]

    for cat, items in largest:
        stats.append(f"  - {cat}: {len(items)} items")

    stats.append("")
    stats.append("**Smallest Categories:**")

    for cat, items in smallest:
        stats.append(f"  - {cat}: {len(items)} items")

    return "\n".join(stats)
