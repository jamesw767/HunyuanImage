#!/usr/bin/env python3
"""
Creative Prompt Generator for HunyuanImage-3.0

Uses Ollama LLMs to generate themed prompt collections, variations,
and creative prompt ideas.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import logging

from ollama_prompts import PromptEnhancer, OllamaClient, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path("/media/james/DataDrive/jamesw767/Hun3d")
OUTPUT_DIR = BASE_DIR / "outputs"


# Specialized generation prompts
THEME_GENERATION_PROMPT = """You are a creative director for a high-end image generation studio.

Generate {count} unique, detailed image prompts for the theme: "{theme}"

Each prompt should:
- Be a complete, standalone scene description
- Include rich visual details (subjects, settings, lighting, mood)
- Be 50-150 words long
- Be unique and different from other prompts
- Follow cinematographic principles

{style_instruction}

Output ONLY the prompts, one per paragraph (separated by blank lines), no numbering or explanations."""

CHARACTER_CONSISTENCY_PROMPT = """You are an expert at creating character-consistent image prompts.

Base character description:
{character}

Generate {count} different scenes featuring this exact character:
- Keep the character's physical appearance EXACTLY the same in each prompt
- Change only: setting, action, lighting, camera angle, mood
- Include the full character description in each prompt
- Each scene should be visually distinct

Output ONLY the prompts, one per paragraph, no numbering."""

STYLE_FUSION_PROMPT = """You are an art director who excels at blending artistic styles.

Create {count} image prompts that blend these styles:
{styles}

Subject/theme: {theme}

Each prompt should:
- Seamlessly combine elements from all the listed styles
- Create a unique visual fusion
- Be detailed enough for image generation (50-150 words)

Output ONLY the prompts, one per paragraph, no numbering."""

STORY_SEQUENCE_PROMPT = """You are a visual storyteller creating a sequence of connected images.

Story concept: {concept}
Number of scenes: {count}

Create {count} sequential image prompts that tell this story visually:
- Each prompt is one scene in the narrative
- Scenes should flow logically from one to the next
- Maintain visual consistency (same characters, setting style)
- Each prompt should be 50-150 words

Output ONLY the prompts, one per paragraph (in order), no numbering."""


class PromptGenerator:
    """Generate creative prompts using Ollama"""

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = "http://localhost:11434"):
        self.client = OllamaClient(ollama_url, model)
        self.enhancer = PromptEnhancer(ollama_url, model)

    def generate_themed_prompts(
        self,
        theme: str,
        count: int = 10,
        style: Optional[str] = None,
        temperature: float = 0.8,
        length: str = "medium",
        complexity: str = "detailed"
    ) -> List[str]:
        """Generate prompts for a theme"""
        # Use the PromptEnhancer's generate_prompts which supports length/complexity
        return self.enhancer.generate_prompts(
            theme=theme,
            count=count,
            style=style,
            temperature=temperature,
            length=length,
            complexity=complexity
        )

    def generate_character_scenes(
        self,
        character_description: str,
        count: int = 5,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate different scenes featuring the same character"""
        prompt = CHARACTER_CONSISTENCY_PROMPT.format(
            character=character_description,
            count=count
        )

        response = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=count * 250
        )

        prompts = [p.strip() for p in response.text.split('\n\n') if p.strip()]
        logger.info(f"Generated {len(prompts)} character scenes")
        return prompts[:count]

    def generate_style_fusion(
        self,
        styles: List[str],
        theme: str,
        count: int = 5,
        temperature: float = 0.8
    ) -> List[str]:
        """Generate prompts that blend multiple art styles"""
        styles_text = "\n".join(f"- {s}" for s in styles)

        prompt = STYLE_FUSION_PROMPT.format(
            styles=styles_text,
            theme=theme,
            count=count
        )

        response = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=count * 250
        )

        prompts = [p.strip() for p in response.text.split('\n\n') if p.strip()]
        logger.info(f"Generated {len(prompts)} style fusion prompts")
        return prompts[:count]

    def generate_story_sequence(
        self,
        concept: str,
        count: int = 5,
        temperature: float = 0.7
    ) -> List[str]:
        """Generate a sequence of prompts that tell a story"""
        prompt = STORY_SEQUENCE_PROMPT.format(
            concept=concept,
            count=count
        )

        response = self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=count * 250
        )

        prompts = [p.strip() for p in response.text.split('\n\n') if p.strip()]
        logger.info(f"Generated {len(prompts)} story sequence prompts")
        return prompts[:count]

    def enhance_batch(
        self,
        prompts: List[str],
        style: Optional[str] = None,
        temperature: float = 0.7
    ) -> List[str]:
        """Enhance a list of prompts"""
        enhanced = []
        for i, p in enumerate(prompts):
            try:
                e = self.enhancer.enhance(p, style=style, temperature=temperature)
                enhanced.append(e)
                logger.info(f"Enhanced {i+1}/{len(prompts)}")
            except Exception as ex:
                logger.error(f"Failed to enhance prompt {i+1}: {ex}")
                enhanced.append(p)  # Keep original
        return enhanced


def save_prompts_to_file(prompts: List[str], output_path: Path, format: str = "json"):
    """Save prompts to file"""
    if format == "json":
        data = {
            "batch_name": output_path.stem,
            "created_at": datetime.now().isoformat(),
            "prompts": [{"prompt": p} for p in prompts]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    elif format == "txt":
        with open(output_path, 'w') as f:
            for p in prompts:
                f.write(p + "\n\n")

    elif format == "csv":
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['prompt', 'style', 'seed', 'aspect_ratio', 'steps'])
            writer.writeheader()
            for p in prompts:
                writer.writerow({'prompt': p, 'style': 'none', 'seed': '', 'aspect_ratio': '1024x1024', 'steps': 20})

    logger.info(f"Saved {len(prompts)} prompts to {output_path}")


def main():
    """CLI interface for prompt generation"""
    parser = argparse.ArgumentParser(description="Creative Prompt Generator for HunyuanImage-3.0")

    parser.add_argument("theme", nargs="?", help="Theme or concept to generate prompts for")
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of prompts to generate")
    parser.add_argument("--style", "-s", help="Art style preference")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Creativity (0.0-1.0)")

    parser.add_argument("--character", "-c", help="Character description for consistent scenes")
    parser.add_argument("--fusion", nargs="+", help="Styles to fuse (--fusion 'art nouveau' 'cyberpunk')")
    parser.add_argument("--story", help="Story concept for sequential prompts")

    parser.add_argument("--enhance", "-e", action="store_true", help="Enhance generated prompts")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "txt", "csv"], default="json", help="Output format")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON to stdout")

    args = parser.parse_args()

    if not args.theme and not args.character and not args.story:
        parser.error("Please provide a theme, --character, or --story")

    generator = PromptGenerator(model=args.model)
    prompts = []

    try:
        if args.character:
            # Character consistency mode
            prompts = generator.generate_character_scenes(
                args.character,
                count=args.count,
                temperature=args.temperature
            )

        elif args.fusion and args.theme:
            # Style fusion mode
            prompts = generator.generate_style_fusion(
                styles=args.fusion,
                theme=args.theme,
                count=args.count,
                temperature=args.temperature
            )

        elif args.story:
            # Story sequence mode
            prompts = generator.generate_story_sequence(
                concept=args.story,
                count=args.count,
                temperature=args.temperature
            )

        else:
            # Standard theme generation
            prompts = generator.generate_themed_prompts(
                theme=args.theme,
                count=args.count,
                style=args.style,
                temperature=args.temperature
            )

        # Enhance if requested
        if args.enhance:
            print("Enhancing prompts...")
            prompts = generator.enhance_batch(prompts, style=args.style)

        # Output
        if args.output:
            output_path = Path(args.output)
            save_prompts_to_file(prompts, output_path, args.format)
            print(f"Saved {len(prompts)} prompts to {output_path}")

        elif args.json:
            print(json.dumps(prompts))

        else:
            print(f"\nGenerated {len(prompts)} prompts:\n")
            for i, p in enumerate(prompts, 1):
                print(f"[{i}] {p}\n")

    except ConnectionError as e:
        print(f"ERROR: {e}")
        print("Make sure Ollama is running: ollama serve")
        exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
