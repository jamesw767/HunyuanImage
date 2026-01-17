"""
FLUX.2 Klein 9B Engine Wrapper.

Wraps the FLUX.2 Klein 9B model to conform to the BaseEngine interface.
Supports higher resolutions than Hunyuan (up to 3072x3072).
"""

import gc
import time
import random
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from PIL import Image

from .base import (
    BaseEngine, EngineCapabilities, GenerationParams, GenerationResult, EngineRegistry
)


# Default config
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
HF_CACHE = "/media/james/BigDrive/AI/cache/huggingface"

# Resolution presets for FLUX (supports higher resolutions)
FLUX_RESOLUTIONS = {
    "3K Square (3072x3072)": (3072, 3072),
    "3K Landscape (3072x1728)": (3072, 1728),
    "3K Portrait (1728x3072)": (1728, 3072),
    "2K Square (2048x2048)": (2048, 2048),
    "2K Landscape (2048x1152)": (2048, 1152),
    "2K Portrait (1152x2048)": (1152, 2048),
    "QHD Display (2560x1440)": (2560, 1440),
    "QHD Phone (1440x2560)": (1440, 2560),
    "Full HD Display (1920x1088)": (1920, 1088),
    "Full HD Phone (1088x1920)": (1088, 1920),
    "1.5K Square (1536x1536)": (1536, 1536),
    "1K Square (1024x1024)": (1024, 1024),
    "768 Square (768x768)": (768, 768),
}


class FluxKleinEngine(BaseEngine):
    """FLUX.2 Klein 9B engine."""

    def __init__(self, gpu_index: int = 1, model_id: str = DEFAULT_MODEL_ID):
        """
        Initialize FLUX engine.

        Args:
            gpu_index: GPU to use (default 1 = Blackwell)
            model_id: HuggingFace model ID
        """
        super().__init__(gpu_index)
        self.model_id = model_id
        self._caps = None
        self.pipeline = None

    @property
    def capabilities(self) -> EngineCapabilities:
        if self._caps is None:
            self._caps = EngineCapabilities(
                name="flux_klein",
                display_name="FLUX.2 Klein 9B",

                # Resolution - FLUX supports much higher resolutions
                min_width=512,
                max_width=3072,
                min_height=512,
                max_height=3072,
                resolution_step=16,  # FLUX prefers multiples of 16
                default_width=1024,
                default_height=1024,

                # Steps
                min_steps=10,
                max_steps=100,
                default_steps=50,

                # Guidance - FLUX uses CFG
                supports_guidance=True,
                min_guidance=1.0,
                max_guidance=10.0,
                default_guidance=4.0,

                # Features
                supports_negative_prompt=True,
                supports_img2img=True,
                supports_inpainting=False,
                supports_controlnet=False,

                # Memory - 9B model, but high-res needs more VRAM
                vram_required_gb=29.0,
                recommended_gpu="Blackwell (98GB) or RTX 4090",

                resolution_presets=FLUX_RESOLUTIONS,
            )
        return self._caps

    def load(self) -> bool:
        """Load FLUX model into VRAM."""
        import os
        os.environ["HF_HOME"] = HF_CACHE

        with self.load_lock:
            if self.is_loaded:
                return True

            try:
                from diffusers import Flux2KleinPipeline, PipelineQuantizationConfig

                print(f"[FLUX] Loading model {self.model_id}...")
                print(f"[FLUX] Target GPU: cuda:{self.gpu_index}")

                # Quantization config for memory efficiency
                quant_config = PipelineQuantizationConfig(
                    quant_backend="torchao",
                    quant_kwargs={"quantization_type": "fp8dq"},
                    components_to_quantize=["transformer"],
                )

                self.pipeline = Flux2KleinPipeline.from_pretrained(
                    self.model_id,
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                )
                self.pipeline.to(f"cuda:{self.gpu_index}")

                # Enable memory optimizations
                self.pipeline.enable_attention_slicing()

                self.model = self.pipeline  # For compatibility
                self.is_loaded = True
                print("[FLUX] Model loaded successfully!")
                return True

            except Exception as e:
                print(f"[FLUX] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                self.pipeline = None
                self.model = None
                self.is_loaded = False
                return False

    def unload(self) -> bool:
        """Unload FLUX model from VRAM."""
        with self.load_lock:
            if not self.is_loaded:
                return True

            try:
                print("[FLUX] Unloading model...")

                if self.pipeline is not None:
                    del self.pipeline
                    self.pipeline = None
                    self.model = None

                gc.collect()
                torch.cuda.empty_cache()

                self.is_loaded = False
                print("[FLUX] Model unloaded successfully!")
                return True

            except Exception as e:
                print(f"[FLUX] Error unloading model: {e}")
                return False

    def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate image(s) with FLUX."""
        if not self.is_model_loaded():
            return GenerationResult(
                success=False,
                error_message="Model not loaded",
                engine_name="flux_klein"
            )

        # Validate params
        valid, error = self.validate_params(params)
        if not valid:
            return GenerationResult(
                success=False,
                error_message=error,
                engine_name="flux_klein"
            )

        images = []
        seeds = []
        start_time = time.time()

        try:
            for i in range(params.batch_size):
                # Determine seed
                if params.seed >= 0:
                    current_seed = params.seed if i == 0 else params.seed + i
                else:
                    current_seed = random.randint(0, 2**31 - 1)

                # Create generator for reproducibility
                generator = torch.Generator(device=f"cuda:{self.gpu_index}")
                generator.manual_seed(current_seed)

                print(f"[FLUX] Generating image {i+1}/{params.batch_size}, seed={current_seed}")
                print(f"[FLUX] Size: {params.width}x{params.height}, steps={params.steps}, cfg={params.guidance_scale}")

                # Generate
                result = self.pipeline(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt if params.negative_prompt else None,
                    width=params.width,
                    height=params.height,
                    num_inference_steps=params.steps,
                    guidance_scale=params.guidance_scale,
                    generator=generator,
                )

                if result.images:
                    images.append(result.images[0])
                    seeds.append(current_seed)

                # Cleanup between generations
                gc.collect()
                torch.cuda.empty_cache()

            gen_time = time.time() - start_time

            return GenerationResult(
                success=len(images) > 0,
                images=images,
                seeds=seeds,
                generation_time=gen_time,
                params_used=params,
                engine_name="flux_klein"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return GenerationResult(
                success=False,
                error_message=str(e),
                engine_name="flux_klein"
            )


# Register the engine
EngineRegistry.register("flux_klein", FluxKleinEngine)
