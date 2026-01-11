"""
Global shared state for HunyuanImage-3.0 UI.
All UI modules import this for cross-module state access.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from pathlib import Path


@dataclass
class AppState:
    """Central application state container."""

    # Model state
    model: Any = None
    model_loaded: bool = False
    model_load_lock: threading.Lock = field(default_factory=threading.Lock)

    # GPU configuration
    selected_gpu: int = 1  # Default to GPU 1 (Blackwell) for image generation
    selected_ollama_gpu: int = 0  # Default to GPU 0 for Ollama LLM
    available_gpus: List[Dict] = field(default_factory=list)

    # Batch state
    batch_running: bool = False
    batch_stop_requested: bool = False
    batch_results: List[Dict] = field(default_factory=list)

    # Single generation state
    single_generation_stop: bool = False

    # Session state
    current_session_dir: Optional[Path] = None
    session_counter: int = 0

    # Custom batch directory
    custom_batch_base_dir: Optional[str] = None

    # Ollama state
    ollama_available: bool = False
    ollama_enhancer: Any = None
    ollama_generator: Any = None
    ollama_manager: Any = None

    # Wildcard state
    wildcard_available: bool = False
    wildcard_manager: Any = None

    # Style presets (loaded dynamically)
    style_presets: Dict[str, str] = field(default_factory=dict)


# Global singleton instance
app_state = AppState()


def get_state() -> AppState:
    """Get the global application state."""
    return app_state


def detect_gpus() -> List[Dict]:
    """Detect available CUDA GPUs using PyTorch indexing (not nvidia-smi).

    Note: PyTorch GPU indices may differ from nvidia-smi ordering.
    We use PyTorch's view since that's what the model uses.
    """
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    'index': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'display': f"PyTorch GPU {i}: {props.name} ({props.total_memory / (1024**3):.0f} GB)"
                })
    except Exception as e:
        print(f"[GPU] Error detecting GPUs: {e}")
    return gpus


def set_gpu(gpu_index: int) -> str:
    """Set the active GPU for image generation (CUDA operations)."""
    state = get_state()
    state.selected_gpu = gpu_index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Find GPU name for confirmation
    for gpu in state.available_gpus:
        if gpu['index'] == gpu_index:
            return f"Image GPU set to: {gpu['name']}"
    return f"Image GPU set to index {gpu_index}"


def set_ollama_gpu(gpu_index: int) -> str:
    """Set the GPU for Ollama LLM operations."""
    state = get_state()
    state.selected_ollama_gpu = gpu_index

    # Ollama uses CUDA_VISIBLE_DEVICES when it starts
    # We'll pass this when restarting Ollama
    for gpu in state.available_gpus:
        if gpu['index'] == gpu_index:
            return f"Ollama GPU set to: {gpu['name']}"
    return f"Ollama GPU set to index {gpu_index}"


def init_gpus() -> None:
    """Initialize GPU detection and set default."""
    state = get_state()
    state.available_gpus = detect_gpus()

    if state.available_gpus:
        # Default to GPU 1 if available (Blackwell), otherwise GPU 0
        default_idx = 1 if len(state.available_gpus) > 1 else 0
        set_gpu(default_idx)
        print(f"[GPU] Detected {len(state.available_gpus)} GPU(s), using GPU {default_idx}")
    else:
        print("[GPU] No CUDA GPUs detected")
