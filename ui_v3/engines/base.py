"""
Base Engine Interface for Multi-Engine Image Generation.

All rendering engines must implement this interface to be usable in the v3 UI.
This allows hot-swapping between different models (Hunyuan, FLUX, etc.) with
a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from pathlib import Path
import threading


@dataclass
class EngineCapabilities:
    """Describes what an engine can do."""
    name: str
    display_name: str

    # Resolution limits
    min_width: int = 512
    max_width: int = 2048
    min_height: int = 512
    max_height: int = 2048
    resolution_step: int = 64  # Must be divisible by this

    # Default resolution
    default_width: int = 1024
    default_height: int = 1024

    # Steps/quality
    min_steps: int = 1
    max_steps: int = 100
    default_steps: int = 20

    # Guidance scale (CFG)
    supports_guidance: bool = True
    min_guidance: float = 1.0
    max_guidance: float = 20.0
    default_guidance: float = 7.5

    # Features
    supports_negative_prompt: bool = True
    supports_img2img: bool = False
    supports_inpainting: bool = False
    supports_controlnet: bool = False

    # Memory requirements
    vram_required_gb: float = 8.0
    recommended_gpu: str = "any"

    # Resolution presets (name -> (width, height))
    resolution_presets: Dict[str, Tuple[int, int]] = field(default_factory=dict)


@dataclass
class GenerationParams:
    """Parameters for image generation - unified across all engines."""
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    guidance_scale: float = 7.5
    seed: int = -1  # -1 = random
    batch_size: int = 1

    # Optional advanced params
    scheduler: Optional[str] = None
    clip_skip: Optional[int] = None

    # Engine-specific params (passed through)
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from image generation."""
    success: bool
    images: List[Image.Image] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    generation_time: float = 0.0
    error_message: str = ""

    # Metadata
    params_used: Optional[GenerationParams] = None
    engine_name: str = ""


class BaseEngine(ABC):
    """Abstract base class for all image generation engines."""

    def __init__(self, gpu_index: int = 0):
        """
        Initialize engine.

        Args:
            gpu_index: Which GPU to use (0, 1, etc.)
        """
        self.gpu_index = gpu_index
        self.model = None
        self.is_loaded = False
        self.load_lock = threading.Lock()
        self._capabilities: Optional[EngineCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """Return the capabilities of this engine."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """
        Load the model into VRAM.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def unload(self) -> bool:
        """
        Unload the model from VRAM.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def generate(self, params: GenerationParams) -> GenerationResult:
        """
        Generate image(s) with the given parameters.

        Args:
            params: GenerationParams with all settings

        Returns:
            GenerationResult with images or error
        """
        pass

    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self.is_loaded and self.model is not None

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            "name": self.capabilities.name,
            "loaded": self.is_model_loaded(),
            "gpu_index": self.gpu_index,
            "vram_required": self.capabilities.vram_required_gb,
        }

    def validate_params(self, params: GenerationParams) -> Tuple[bool, str]:
        """
        Validate generation parameters against capabilities.

        Returns:
            (is_valid, error_message)
        """
        caps = self.capabilities

        # Check resolution
        if params.width < caps.min_width or params.width > caps.max_width:
            return False, f"Width must be {caps.min_width}-{caps.max_width}"
        if params.height < caps.min_height or params.height > caps.max_height:
            return False, f"Height must be {caps.min_height}-{caps.max_height}"
        if params.width % caps.resolution_step != 0:
            return False, f"Width must be divisible by {caps.resolution_step}"
        if params.height % caps.resolution_step != 0:
            return False, f"Height must be divisible by {caps.resolution_step}"

        # Check steps
        if params.steps < caps.min_steps or params.steps > caps.max_steps:
            return False, f"Steps must be {caps.min_steps}-{caps.max_steps}"

        # Check guidance
        if caps.supports_guidance:
            if params.guidance_scale < caps.min_guidance or params.guidance_scale > caps.max_guidance:
                return False, f"Guidance must be {caps.min_guidance}-{caps.max_guidance}"

        return True, ""


class EngineRegistry:
    """Registry of available engines."""

    _engines: Dict[str, type] = {}
    _instances: Dict[str, BaseEngine] = {}

    @classmethod
    def register(cls, name: str, engine_class: type):
        """Register an engine class."""
        cls._engines[name] = engine_class

    @classmethod
    def get_available_engines(cls) -> List[str]:
        """Get list of registered engine names."""
        return list(cls._engines.keys())

    @classmethod
    def create_engine(cls, name: str, gpu_index: int = 0) -> Optional[BaseEngine]:
        """Create an instance of an engine."""
        if name not in cls._engines:
            return None
        return cls._engines[name](gpu_index=gpu_index)

    @classmethod
    def get_or_create(cls, name: str, gpu_index: int = 0) -> Optional[BaseEngine]:
        """Get existing instance or create new one."""
        key = f"{name}_{gpu_index}"
        if key not in cls._instances:
            engine = cls.create_engine(name, gpu_index)
            if engine:
                cls._instances[key] = engine
        return cls._instances.get(key)

    @classmethod
    def get_all_capabilities(cls) -> Dict[str, EngineCapabilities]:
        """Get capabilities of all registered engines."""
        caps = {}
        for name, engine_class in cls._engines.items():
            # Create temporary instance to get capabilities
            try:
                temp = engine_class(gpu_index=0)
                caps[name] = temp.capabilities
            except Exception:
                pass
        return caps
