#!/usr/bin/env python3
"""
Wildcard Manager for HunyuanImage-3.0

Handles loading of wildcard definitions from JSON and random substitution
in prompts. Wildcards are specified using [key] syntax.

Example:
    Prompt: "A [animal] in a [landscape] setting"
    Result: "A tiger in a forest setting"
"""

import json
import random
import re
import os
from pathlib import Path
from typing import List, Dict, Optional


class WildcardManager:
    def __init__(self, json_path: str = "wildcards.json"):
        self.json_path = Path(json_path)
        self.data: Dict[str, List[str]] = {}
        self._load_data()

    def _load_data(self):
        """Load wildcard definitions from JSON file"""
        if not self.json_path.exists():
            print(f"Warning: {self.json_path} not found. Wildcards will not work.")
            return

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} wildcard categories")
        except Exception as e:
            print(f"Error loading wildcards: {e}")
            self.data = {}

    def reload(self):
        """Reload wildcards from file"""
        self._load_data()
        return f"Reloaded {len(self.data)} wildcard categories"

    def get_available_wildcards(self) -> List[str]:
        """Returns a sorted list of all wildcard keys"""
        return sorted(list(self.data.keys()))

    def get_wildcard_count(self, key: str) -> int:
        """Get the number of items in a wildcard category"""
        return len(self.data.get(key, []))

    def get_wildcard_preview(self, key: str, count: int = 5) -> str:
        """Get a preview of items in a wildcard category"""
        items = self.data.get(key, [])
        if not items:
            return f"[{key}] - Not found"

        preview = items[:count]
        remaining = len(items) - count
        preview_str = ", ".join(preview)
        if remaining > 0:
            preview_str += f" ... and {remaining} more"
        return f"[{key}] ({len(items)} items): {preview_str}"

    def get_random_value(self, key: str) -> Optional[str]:
        """Get a random value from a wildcard category"""
        items = self.data.get(key, [])
        if items:
            return random.choice(items)
        return None

    def process_prompt(self, prompt: str, seed: Optional[int] = None) -> str:
        """
        Process a prompt and replace all [key] wildcards with random values.

        Args:
            prompt: The prompt text with [wildcard] placeholders
            seed: Optional random seed for reproducibility

        Returns:
            The processed prompt with wildcards replaced
        """
        if not prompt:
            return ""

        # Set seed if provided for reproducible results
        if seed is not None:
            random.seed(seed)

        # Regex to find [something] or [ something ]
        pattern = r"\[\s*([^\]]+?)\s*\]"

        def replace_match(match):
            key = match.group(1).strip()

            # Check if key exists in our data
            if key in self.data:
                choice = random.choice(self.data[key])
                return choice
            else:
                # Key not found, keep original to avoid breaking user prompt
                return match.group(0)

        # Keep replacing until no more known wildcards are found
        # (handles nested wildcards if a value contains another [wildcard])
        previous_prompt = ""
        max_iterations = 10  # Prevent infinite loops
        iterations = 0

        while prompt != previous_prompt and iterations < max_iterations:
            previous_prompt = prompt
            prompt = re.sub(pattern, replace_match, prompt)
            iterations += 1

        # Reset random seed to avoid affecting other random operations
        if seed is not None:
            random.seed()

        return prompt

    def process_prompt_batch(self, prompt: str, count: int,
                             base_seed: Optional[int] = None) -> List[str]:
        """
        Generate multiple variations of a prompt with different wildcard values.

        Args:
            prompt: The prompt text with [wildcard] placeholders
            count: Number of variations to generate
            base_seed: Optional base seed (each variation uses base_seed + index)

        Returns:
            List of processed prompts with different random values
        """
        results = []
        for i in range(count):
            seed = (base_seed + i) if base_seed is not None else None
            processed = self.process_prompt(prompt, seed=seed)
            results.append(processed)
        return results

    def has_wildcards(self, prompt: str) -> bool:
        """Check if a prompt contains any wildcard syntax"""
        if not prompt:
            return False
        pattern = r"\[\s*([^\]]+?)\s*\]"
        matches = re.findall(pattern, prompt)
        # Check if any of the matches are valid wildcard keys
        return any(key.strip() in self.data for key in matches)

    def list_wildcards_in_prompt(self, prompt: str) -> List[str]:
        """List all wildcards used in a prompt"""
        if not prompt:
            return []
        pattern = r"\[\s*([^\]]+?)\s*\]"
        matches = re.findall(pattern, prompt)
        return [key.strip() for key in matches if key.strip() in self.data]

    def get_categories(self) -> Dict[str, List[str]]:
        """Group wildcards by category prefix"""
        categories = {}
        for key in self.data.keys():
            parts = key.split('-')
            if len(parts) > 1:
                category = parts[0]
            else:
                category = "general"

            if category not in categories:
                categories[category] = []
            categories[category].append(key)

        # Sort each category
        for cat in categories:
            categories[cat] = sorted(categories[cat])

        return categories

    def search_wildcards(self, query: str) -> List[str]:
        """Search for wildcards matching a query"""
        query = query.lower()
        return [key for key in self.data.keys() if query in key.lower()]


# Global instance for easy import
wildcard_manager = WildcardManager(
    json_path=Path(__file__).parent / "wildcards.json"
)


def insert_wildcard(current_prompt: str, selected_wildcard: str) -> str:
    """Insert a wildcard tag into the prompt"""
    if not selected_wildcard:
        return current_prompt

    tag = f"[{selected_wildcard}]"
    if current_prompt:
        # Add space before if needed
        if not current_prompt.endswith(' '):
            return current_prompt + ' ' + tag
        return current_prompt + tag
    return tag


def preview_wildcard(key: str) -> str:
    """Get a preview of a wildcard category"""
    return wildcard_manager.get_wildcard_preview(key, count=8)


if __name__ == "__main__":
    # Test the wildcard manager
    print("Testing Wildcard Manager")
    print("=" * 50)

    print(f"\nAvailable wildcards: {len(wildcard_manager.get_available_wildcards())}")

    # Show some categories
    categories = wildcard_manager.get_categories()
    print(f"\nCategories: {list(categories.keys())[:10]}...")

    # Test prompt processing
    test_prompt = "A [animal] standing in a [landscape] with [weather] weather"
    print(f"\nTest prompt: {test_prompt}")

    for i in range(3):
        result = wildcard_manager.process_prompt(test_prompt)
        print(f"  Variation {i+1}: {result}")

    # Test with seed for reproducibility
    print("\nWith seed=42:")
    for i in range(2):
        result = wildcard_manager.process_prompt(test_prompt, seed=42)
        print(f"  Run {i+1}: {result}")
