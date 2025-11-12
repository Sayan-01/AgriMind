"""
Simple configuration for AgriMind ML Inference
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Model configuration
MODEL_NAME = "rexnet_150"
NUM_CLASSES = 17  # Updated to match the trained model
INPUT_SIZE = (224, 224)

# Default model path
DEFAULT_MODEL_PATH = OUTPUT_DIR / "crop_best_model.pth"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
