"""
HunyuanImage-3.0 Engine Wrapper.

Wraps the 80B MoE quantized HunyuanImage model to conform to the BaseEngine interface.
"""

import gc
import time
import random
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import torch
from PIL import Image

from .base import (
    BaseEngine, EngineCapabilities, GenerationParams, GenerationResult, EngineRegistry
)


# Default paths
DEFAULT_MODEL_PATH = Path("/media/james/DataDrive/jamesw767/Hun3d/HunyuanImage3-SDNQ")

# Resolution presets for Hunyuan (limited to 1024 max recommended)
HUNYUAN_RESOLUTIONS = {
    "1:1 Square (1024x1024)": (1024, 1024),
    "16:9 Landscape (1536x864)": (1536, 864),
    "9:16 Portrait (864x1536)": (864, 1536),
    "4:3 (1152x896)": (1152, 896),
    "3:4 (896x1152)": (896, 1152),
    "3:2 (1216x832)": (1216, 832),
    "2:3 (832x1216)": (832, 1216),
    "21:9 Ultrawide (1680x720)": (1680, 720),
}


class HunyuanEngine(BaseEngine):
    """HunyuanImage-3.0 80B MoE engine."""

    def __init__(self, gpu_index: int = 1, model_path: Optional[Path] = None):
        """
        Initialize Hunyuan engine.

        Args:
            gpu_index: GPU to use (default 1 = Blackwell)
            model_path: Path to model weights
        """
        super().__init__(gpu_index)
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._caps = None

    @property
    def capabilities(self) -> EngineCapabilities:
        if self._caps is None:
            self._caps = EngineCapabilities(
                name="hunyuan",
                display_name="HunyuanImage-3.0 (80B MoE)",

                # Resolution - Hunyuan works best at 1024, can go to 1536
                min_width=512,
                max_width=1536,
                min_height=512,
                max_height=1536,
                resolution_step=64,
                default_width=1024,
                default_height=1024,

                # Steps - Hunyuan uses diff_infer_steps
                min_steps=10,
                max_steps=100,
                default_steps=20,

                # Guidance - Hunyuan doesn't use traditional CFG
                supports_guidance=False,
                default_guidance=1.0,

                # Features
                supports_negative_prompt=False,  # Hunyuan doesn't use negative prompts
                supports_img2img=False,
                supports_inpainting=False,
                supports_controlnet=False,

                # Memory - 80B MoE needs significant VRAM
                vram_required_gb=48.0,
                recommended_gpu="Blackwell (98GB)",

                resolution_presets=HUNYUAN_RESOLUTIONS,
            )
        return self._caps

    def load(self) -> bool:
        """Load Hunyuan model into VRAM."""
        with self.load_lock:
            if self.is_loaded:
                return True

            try:
                from transformers import AutoModelForCausalLM

                print(f"[HUNYUAN] Loading model from {self.model_path}...")
                print(f"[HUNYUAN] Target GPU: cuda:{self.gpu_index}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    attn_implementation="sdpa",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=f"cuda:{self.gpu_index}",
                    moe_impl="eager",
                )
                self.model.load_tokenizer(str(self.model_path))

                self.is_loaded = True
                print("[HUNYUAN] Model loaded successfully!")
                return True

            except Exception as e:
                print(f"[HUNYUAN] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
                self.is_loaded = False
                return False

    def unload(self) -> bool:
        """Unload Hunyuan model from VRAM."""
        with self.load_lock:
            if not self.is_loaded:
                return True

            try:
                print("[HUNYUAN] Unloading model...")

                if self.model is not None:
                    del self.model
                    self.model = None

                gc.collect()
                torch.cuda.empty_cache()

                self.is_loaded = False
                print("[HUNYUAN] Model unloaded successfully!")
                return True

            except Exception as e:
                print(f"[HUNYUAN] Error unloading model: {e}")
                return False

    def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate image(s) with Hunyuan."""
        if not self.is_model_loaded():
            return GenerationResult(
                success=False,
                error_message="Model not loaded",
                engine_name="hunyuan"
            )

        # Validate params
        valid, error = self.validate_params(params)
        if not valid:
            return GenerationResult(
                success=False,
                error_message=error,
                engine_name="hunyuan"
            )

        images = []
        seeds = []
        start_time = time.time()

        try:
            # Format resolution as "WxH"
            image_size = f"{params.width}x{params.height}"

            for i in range(params.batch_size):
                # Determine seed
                if params.seed >= 0:
                    current_seed = params.seed if i == 0 else params.seed + i
                else:
                    current_seed = random.randint(0, 2**31 - 1)

                print(f"[HUNYUAN] Generating image {i+1}/{params.batch_size}, seed={current_seed}")

                # Generate
                image = self.model.generate_image(
                    params.prompt,
                    current_seed,
                    image_size,
                    stream=True,
                    diff_infer_steps=params.steps
                )

                if image is not None:
                    images.append(image)
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
                engine_name="hunyuan"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return GenerationResult(
                success=False,
                error_message=str(e),
                engine_name="hunyuan"
            )


# Register the engine
EngineRegistry.register("hunyuan", HunyuanEngine)
