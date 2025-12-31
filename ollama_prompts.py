#!/usr/bin/env python3
"""
Ollama Prompt Enhancement System for HunyuanImage-3.0

Provides local LLM-based prompt enhancement, generation, and batch processing
without requiring external API keys.
"""

import json
import requests
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

# System prompts following HunyuanImage's cinematographic formula
ENHANCE_SYSTEM_PROMPT = """You are an expert image prompt engineer for HunyuanImage-3.0, an advanced AI image generator.

Your task is to enhance the user's simple prompt into a detailed, cinematic description that will generate stunning images. Follow this 5-part cinematographic formula:

1. **Main Subject & Scene**: Describe the primary subject with rich, specific details (materials, textures, expressions, poses, environment context)
2. **Image Quality & Style**: Add quality descriptors (8K resolution, photorealistic, hyperdetailed, masterpiece quality)
3. **Composition & Viewpoint**: Specify camera angle (low angle, bird's eye, close-up), framing, depth of field, perspective
4. **Lighting & Atmosphere**: Define lighting (golden hour, dramatic shadows, soft diffused light), mood, time of day, weather
5. **Technical Parameters**: Mention art style or rendering technique when appropriate (cinematic, oil painting, digital art, etc.)

RULES:
- Keep the core intent and subject of the original prompt
- Output ONLY the enhanced prompt text, no explanations or labels
- Keep it under 200 words for optimal results
- Be creative but stay true to what was requested
- Don't add inappropriate or unsafe content"""

GENERATE_PROMPTS_SYSTEM = """You are a creative prompt generator for HunyuanImage-3.0 image generation.

Given a theme or concept, generate unique, detailed image prompts. Each prompt should be:
- Self-contained and complete
- Visually interesting and specific
- Between 50-150 words
- Following cinematographic principles (subject, quality, composition, lighting, style)

Output ONLY the prompts, one per line, no numbering or explanations."""

VARIATION_SYSTEM_PROMPT = """You are a creative prompt variation generator for HunyuanImage-3.0.

Given a base prompt, create variations that:
- Maintain the core subject/concept
- Change aspects like: angle, lighting, time of day, style, mood, setting
- Are distinctly different from each other
- Each stands alone as a complete, detailed prompt

Output ONLY the variation prompts, one per line, no numbering or explanations."""

ANALYZE_PROMPT_SYSTEM = """You are an expert image prompt analyst for HunyuanImage-3.0.

Analyze the given prompt and provide:
1. **Strengths**: What's good about this prompt
2. **Missing Elements**: What cinematographic elements are missing
3. **Suggestions**: Specific improvements to make the image better
4. **Enhanced Version**: A rewritten, improved version

Be concise and actionable."""


@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    text: str
    model: str
    total_duration: float  # in seconds
    prompt_eval_count: int
    eval_count: int

    @property
    def tokens_per_second(self) -> float:
        if self.total_duration > 0:
            return self.eval_count / self.total_duration
        return 0.0


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._check_connection()

    def _check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama server not running. Start it with: ollama serve")
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> OllamaResponse:
        """Generate text using Ollama"""
        model = model or self.model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # 5 min timeout for large models
            )

            if response.status_code == 200:
                data = response.json()
                return OllamaResponse(
                    text=data.get('response', '').strip(),
                    model=data.get('model', model),
                    total_duration=data.get('total_duration', 0) / 1e9,  # ns to s
                    prompt_eval_count=data.get('prompt_eval_count', 0),
                    eval_count=data.get('eval_count', 0)
                )
            else:
                raise Exception(f"Ollama error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to Ollama. Is the server running? (ollama serve)")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream generation from Ollama"""
        model = model or self.model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300
            ) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to Ollama. Is the server running?")


class PromptEnhancer:
    """Enhance and generate image prompts using Ollama"""

    def __init__(self, ollama_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.client = OllamaClient(ollama_url, model)

    def enhance(
        self,
        prompt: str,
        style: Optional[str] = None,
        temperature: float = 0.7,
        model: Optional[str] = None
    ) -> str:
        """
        Enhance a simple prompt into a detailed, cinematic description.

        Args:
            prompt: The simple prompt to enhance
            style: Optional style to incorporate (e.g., "cinematic", "anime", "oil painting")
            temperature: Creativity level (0.0-1.0)
            model: Ollama model to use (default: qwen2.5:7b-instruct)

        Returns:
            Enhanced prompt string
        """
        user_prompt = prompt
        if style:
            user_prompt = f"[Style: {style}] {prompt}"

        response = self.client.generate(
            prompt=user_prompt,
            system=ENHANCE_SYSTEM_PROMPT,
            temperature=temperature,
            model=model,
            max_tokens=512
        )

        logger.info(f"Enhanced prompt in {response.total_duration:.1f}s ({response.tokens_per_second:.1f} tok/s)")
        return response.text

    def generate_prompts(
        self,
        theme: str,
        count: int = 10,
        style: Optional[str] = None,
        temperature: float = 0.8,
        model: Optional[str] = None
    ) -> List[str]:
        """
        Generate multiple creative prompts for a theme.

        Args:
            theme: The theme or concept to generate prompts for
            count: Number of prompts to generate
            style: Optional style preference
            temperature: Creativity level
            model: Ollama model to use

        Returns:
            List of generated prompts
        """
        user_prompt = f"Generate {count} unique, detailed image prompts for the theme: '{theme}'"
        if style:
            user_prompt += f"\nPreferred style: {style}"

        response = self.client.generate(
            prompt=user_prompt,
            system=GENERATE_PROMPTS_SYSTEM,
            temperature=temperature,
            model=model,
            max_tokens=count * 200  # ~200 tokens per prompt
        )

        # Parse prompts from response
        prompts = [p.strip() for p in response.text.split('\n') if p.strip()]
        logger.info(f"Generated {len(prompts)} prompts in {response.total_duration:.1f}s")
        return prompts

    def create_variations(
        self,
        base_prompt: str,
        count: int = 5,
        temperature: float = 0.8,
        model: Optional[str] = None
    ) -> List[str]:
        """
        Create variations of a base prompt.

        Args:
            base_prompt: The original prompt to create variations of
            count: Number of variations to generate
            temperature: Creativity level
            model: Ollama model to use

        Returns:
            List of prompt variations
        """
        user_prompt = f"Create {count} distinct variations of this prompt:\n\n{base_prompt}"

        response = self.client.generate(
            prompt=user_prompt,
            system=VARIATION_SYSTEM_PROMPT,
            temperature=temperature,
            model=model,
            max_tokens=count * 200
        )

        variations = [p.strip() for p in response.text.split('\n') if p.strip()]
        logger.info(f"Created {len(variations)} variations in {response.total_duration:.1f}s")
        return variations

    def analyze(
        self,
        prompt: str,
        model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Analyze a prompt and provide improvement suggestions.

        Args:
            prompt: The prompt to analyze
            model: Ollama model to use

        Returns:
            Dict with analysis sections
        """
        response = self.client.generate(
            prompt=f"Analyze this image prompt:\n\n{prompt}",
            system=ANALYZE_PROMPT_SYSTEM,
            temperature=0.5,
            model=model,
            max_tokens=800
        )

        # Return raw analysis text
        return {"analysis": response.text}

    def batch_enhance(
        self,
        prompts: List[str],
        style: Optional[str] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        progress_callback=None
    ) -> List[Dict[str, str]]:
        """
        Enhance multiple prompts.

        Args:
            prompts: List of prompts to enhance
            style: Optional style for all prompts
            temperature: Creativity level
            model: Ollama model to use
            progress_callback: Optional callback(current, total, prompt)

        Returns:
            List of dicts with 'original' and 'enhanced' keys
        """
        results = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            if progress_callback:
                progress_callback(i + 1, total, prompt)

            try:
                enhanced = self.enhance(prompt, style=style, temperature=temperature, model=model)
                results.append({
                    "original": prompt,
                    "enhanced": enhanced
                })
            except Exception as e:
                logger.error(f"Error enhancing prompt {i+1}: {e}")
                results.append({
                    "original": prompt,
                    "enhanced": prompt,  # Fall back to original
                    "error": str(e)
                })

        return results


def main():
    """CLI interface for prompt enhancement"""
    import argparse

    parser = argparse.ArgumentParser(description="Ollama Prompt Enhancement for HunyuanImage-3.0")
    parser.add_argument("prompt", nargs="?", help="Prompt to enhance")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--style", "-s", help="Style to apply (cinematic, anime, etc.)")
    parser.add_argument("--generate", "-g", type=int, metavar="N", help="Generate N prompts for a theme")
    parser.add_argument("--variations", "-v", type=int, metavar="N", help="Create N variations of prompt")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze the prompt")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Temperature (0.0-1.0)")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama server URL")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    enhancer = PromptEnhancer(ollama_url=args.url, model=args.model)

    if args.list_models:
        models = enhancer.client.list_models()
        if args.json:
            print(json.dumps(models))
        else:
            print("Available Ollama models:")
            for m in models:
                print(f"  - {m}")
        return

    if not args.prompt:
        parser.error("Prompt is required (unless using --list-models)")

    try:
        if args.generate:
            # Generate prompts for a theme
            prompts = enhancer.generate_prompts(
                args.prompt,
                count=args.generate,
                style=args.style,
                temperature=args.temperature
            )
            if args.json:
                print(json.dumps(prompts))
            else:
                for i, p in enumerate(prompts, 1):
                    print(f"\n[{i}] {p}")

        elif args.variations:
            # Create variations
            variations = enhancer.create_variations(
                args.prompt,
                count=args.variations,
                temperature=args.temperature
            )
            if args.json:
                print(json.dumps(variations))
            else:
                print(f"\nVariations of: {args.prompt}\n")
                for i, v in enumerate(variations, 1):
                    print(f"[{i}] {v}\n")

        elif args.analyze:
            # Analyze prompt
            analysis = enhancer.analyze(args.prompt)
            if args.json:
                print(json.dumps(analysis))
            else:
                print(analysis["analysis"])

        else:
            # Default: enhance prompt
            enhanced = enhancer.enhance(
                args.prompt,
                style=args.style,
                temperature=args.temperature
            )
            if args.json:
                print(json.dumps({"original": args.prompt, "enhanced": enhanced}))
            else:
                print(enhanced)

    except ConnectionError as e:
        logger.error(str(e))
        print("\nERROR: Cannot connect to Ollama server.")
        print("Start it with: ollama serve")
        exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
