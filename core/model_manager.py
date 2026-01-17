"""
Model management for HunyuanImage-3.0.
Handles loading, unloading, and status of the image generation model.
"""

import torch
from typing import Generator

from ui.state import get_state
from ui.constants import MODEL_PATH


def load_model() -> Generator[str, None, None]:
    """Load the quantized HunyuanImage-3.0 model.

    Yields status messages during loading process.
    """
    state = get_state()

    # Quick check without lock
    if state.model_loaded and state.model is not None:
        yield "Model already loaded and ready!"
        return

    # Try to acquire lock (non-blocking to give feedback)
    if not state.model_load_lock.acquire(blocking=False):
        yield "Model is currently being loaded by another request..."
        # Wait for the other load to complete
        state.model_load_lock.acquire()
        state.model_load_lock.release()
        if state.model_loaded:
            yield "Model loaded by another request. Ready!"
        return

    try:
        # Double-check after acquiring lock
        if state.model_loaded and state.model is not None:
            yield "Model already loaded and ready!"
            return

        # Check if GPU already has significant memory usage
        gpu_idx = state.selected_gpu
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
            if mem_used > 30:  # More than 30GB suggests model is already loaded
                yield f"WARNING: GPU {gpu_idx} already has {mem_used:.1f}GB allocated. Model may already be loaded."
                yield "If you want to reload, unload first."
                return

        print("[LOAD] Step 1: Importing transformers...")
        yield "Step 1: Importing libraries..."
        from transformers import AutoModelForCausalLM

        print("[LOAD] Step 2: Importing SDNQ...")
        from sdnq import SDNQConfig  # Registers SDNQ into transformers

        model_id = str(MODEL_PATH)

        # Get selected GPU from state
        gpu_index = state.selected_gpu
        device = f"cuda:{gpu_index}"

        print(f"[LOAD] Step 3: Loading model from {model_id} to {device}...")
        yield f"Step 2: Loading quantized HunyuanImage-3.0 on GPU {gpu_index}... (this may take 1-2 minutes)"

        state.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            moe_impl="eager",
            local_files_only=True,
        )

        print("[LOAD] Step 4: Model loaded successfully!")
        yield "Step 3: Model weights loaded, setting up tokenizer..."

        # Model is loaded - set flag FIRST in case tokenizer fails
        state.model_loaded = True

        # Try to load tokenizer
        print("[LOAD] Step 5: Loading tokenizer...")
        try:
            state.model.load_tokenizer(model_id, local_files_only=True)
            print("[LOAD] Step 5: Tokenizer loaded successfully!")
        except Exception as tok_err:
            print(f"[LOAD] Tokenizer load warning (model still works): {tok_err}")
            try:
                state.model.load_tokenizer(model_id)
                print("[LOAD] Step 5: Tokenizer loaded (fallback method)")
            except Exception:
                print("[LOAD] Tokenizer fallback also failed - model may still work for generation")

        # Show GPU memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            yield f"Model loaded! Using {mem_used:.1f}GB / {mem_total:.1f}GB GPU memory. Ready to generate."
        else:
            yield "Model loaded successfully! Ready to generate images."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[LOAD ERROR] {str(e)}")
        print(f"[LOAD TRACEBACK]\n{error_details}")
        yield f"Error loading model: {str(e)}\n\nDetails:\n{error_details}"
    finally:
        state.model_load_lock.release()


def unload_model() -> str:
    """Unload the image generation model to free GPU memory."""
    state = get_state()

    if not state.model_loaded or state.model is None:
        return "Model not loaded - nothing to unload"

    try:
        # Delete model and clear CUDA cache
        del state.model
        state.model = None
        state.model_loaded = False

        # Clear CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Report freed memory
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            return f"Model unloaded! GPU memory freed. Current usage: {mem_used:.1f}GB"

        return "Model unloaded successfully!"

    except Exception as e:
        return f"Error unloading model: {str(e)}"


def get_model_status() -> str:
    """Get current model loading status and GPU memory info."""
    state = get_state()

    status_lines = []

    # Image model status
    gpu_idx = state.selected_gpu

    if state.model_loaded and state.model is not None:
        status_lines.append(f"**Image Model: LOADED** (GPU {gpu_idx})")
    else:
        status_lines.append(f"**Image Model: NOT LOADED** (will use GPU {gpu_idx})")

    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            mem_used = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
            mem_total = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)

            status_lines.append(f"GPU {gpu_idx}: {gpu_name}")
            status_lines.append(f"Memory: {mem_used:.1f}GB / {mem_total:.1f}GB")

            if not state.model_loaded:
                status_lines.append("*Ready to load*")
        except Exception as e:
            status_lines.append(f"GPU info error: {e}")
    else:
        status_lines.append("No CUDA GPU available")

    # LLM/Ollama status
    status_lines.append("")  # Separator
    llm_gpu_idx = state.selected_ollama_gpu

    if state.ollama_available and state.ollama_manager:
        try:
            if state.ollama_manager.is_running():
                models = state.ollama_manager.list_models()
                if models:
                    model_names = [m['name'] for m in models[:3]]  # Show first 3
                    models_str = ", ".join(model_names)
                    if len(models) > 3:
                        models_str += f" (+{len(models)-3} more)"
                    status_lines.append(f"**LLM: RUNNING** (GPU {llm_gpu_idx})")
                    status_lines.append(f"Models: {models_str}")
                else:
                    status_lines.append(f"**LLM: RUNNING** (GPU {llm_gpu_idx}, no models)")
            else:
                status_lines.append(f"**LLM: NOT RUNNING** (will use GPU {llm_gpu_idx})")
                status_lines.append("*Start with: ollama serve*")
        except Exception as e:
            status_lines.append(f"**LLM: ERROR** - {e}")
    else:
        status_lines.append("**LLM: NOT AVAILABLE**")

    return "\n".join(status_lines)


def get_model():
    """Get the loaded model instance."""
    state = get_state()
    return state.model


def is_model_loaded() -> bool:
    """Check if the model is currently loaded."""
    state = get_state()
    return state.model_loaded and state.model is not None
