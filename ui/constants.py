"""
Constants and configuration presets for HunyuanImage-3.0 UI.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Base paths
PROJECT_DIR = Path("/media/james/DataDrive/jamesw767/Hun3d")
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = PROJECT_DIR / "HunyuanImage3-SDNQ"

# Config files
STYLE_PRESETS_FILE = OUTPUT_DIR / "style_presets.json"
HISTORY_FILE = OUTPUT_DIR / "generation_history.json"
UI_CONFIG_FILE = PROJECT_DIR / "ui_config.json"
WILDCARDS_FILE = PROJECT_DIR / "wildcards.json"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# Image Generation Settings
# =============================================================================

IMAGE_SIZES = [
    "auto",
    "1024x1024",
    "1280x768",
    "768x1280",
    "1152x896",
    "896x1152",
    "1344x768",
    "768x1344",
    "1536x640",
    "640x1536",
    "1280x720",
    "832x480",
]

ASPECT_RATIOS = {
    "1:1 (Square)": "1024x1024",
    "16:9 (Landscape)": "1280x768",
    "16:9 (720p Video)": "1280x720",
    "16:9 (480p Video)": "832x480",
    "9:16 (Portrait)": "768x1280",
    "4:3 (Standard)": "1152x896",
    "3:4 (Portrait Standard)": "896x1152",
    "21:9 (Ultrawide)": "1536x640",
    "9:21 (Tall)": "640x1536",
    "Auto (Model decides)": "auto",
}

QUALITY_PRESETS = {
    "Draft (Fast)": {"steps": 15, "description": "Quick preview, lower quality"},
    "Standard": {"steps": 20, "description": "Good balance of speed and quality"},
    "High Quality": {"steps": 30, "description": "Better details, slower"},
    "Maximum": {"steps": 50, "description": "Best quality, slowest"},
}

# =============================================================================
# Style Presets
# =============================================================================

DEFAULT_STYLE_PRESETS = {
    "None": "",
    "Photorealistic": ", photorealistic, hyperrealistic, 8k, highly detailed, professional photography",
    "Cinematic": ", cinematic lighting, dramatic atmosphere, movie still, 35mm film grain",
    "Anime": ", anime style, vibrant colors, detailed linework, studio quality anime",
    "Digital Art": ", digital art, concept art, artstation trending, highly detailed illustration",
    "Oil Painting": ", oil painting style, classical art, rich textures, museum quality",
    "Watercolor": ", watercolor painting, soft edges, flowing colors, artistic",
    "3D Render": ", 3D render, octane render, unreal engine 5, high quality CGI",
    "Fantasy Art": ", fantasy art, epic fantasy, magical atmosphere, detailed fantasy illustration",
    "Minimalist": ", minimalist style, clean lines, simple composition, elegant",
    "Vintage": ", vintage photograph, retro style, nostalgic, aged film quality",
    "Comic Book": ", comic book style, bold lines, dynamic composition, graphic novel art",
    "Studio Portrait": ", studio portrait photography, professional lighting, sharp focus, bokeh background",
    "Nature Photography": ", national geographic style, nature photography, stunning natural light",
}

# =============================================================================
# Ollama Settings
# =============================================================================

# Fallback model list (used if Ollama server not running at startup)
OLLAMA_MODELS = ["qwen2.5:7b-instruct"]

# Default Ollama model
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"

# Prompt enhancement options
OLLAMA_LENGTH_OPTIONS = ["minimal", "short", "medium", "long", "detailed"]
OLLAMA_COMPLEXITY_OPTIONS = ["simple", "basic", "moderate", "detailed", "complex"]

# =============================================================================
# UI Configuration
# =============================================================================

# Default UI colors (can be overridden by ui_config.json)
DEFAULT_UI_COLORS = {
    "background": "#1a1a1a",
    "text": "#e0e0e0",
    "primary": "#4a90d9",
    "secondary": "#2d2d2d",
    "accent": "#66bb6a",
    "error": "#ef5350",
    "button_primary": "#4a90d9",
    "button_secondary": "#424242",
    "button_danger": "#ef5350",
}

# =============================================================================
# Generation Defaults
# =============================================================================

DEFAULT_SEED = -1  # -1 means random
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_ASPECT_RATIO = "1:1 (Square)"
DEFAULT_QUALITY = "Standard"
