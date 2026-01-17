"""
Multi-Engine Image Generation System.

Provides a unified interface for different image generation models:
- HunyuanImage-3.0 (80B MoE) - Best quality, 1024px max recommended
- FLUX.2 Klein 9B - High resolution support up to 3072px

Usage:
    from ui_v3.engines import EngineRegistry, GenerationParams

    # Get available engines
    engines = EngineRegistry.get_available_engines()

    # Create and use an engine
    engine = EngineRegistry.get_or_create("hunyuan", gpu_index=1)
    engine.load()

    result = engine.generate(GenerationParams(
        prompt="A beautiful sunset",
        width=1024,
        height=1024,
        steps=20
    ))
"""

from .base import (
    BaseEngine,
    EngineCapabilities,
    GenerationParams,
    GenerationResult,
    EngineRegistry,
)

# Import engines to register them
from . import hunyuan
from . import flux

__all__ = [
    "BaseEngine",
    "EngineCapabilities",
    "GenerationParams",
    "GenerationResult",
    "EngineRegistry",
]
