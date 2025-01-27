from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw" / "training_traces.jsonl"
GENERATED_RESPONSES_PATH = DATA_DIR / "generated" / "epoch1_responses.jsonl"
QUALITY_SCORES_PATH = DATA_DIR / "processed" / "quality_scores.jsonl"

# Model paths
EPOCH1_MODEL_PATH = MODELS_DIR / "epoch1" / "qwen-sft-lora-final"
EPOCH2_MODEL_PATH = MODELS_DIR / "epoch2" / "qwen-sft-lora-epoch2"

# Model config
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 2048
BATCH_SIZE = 1

# Quality evaluation config
QUALITY_THRESHOLD = 0.7  # Minimum score for high-quality samples
METRICS = {
    "thinking_structure": 0.3,  # Weight for proper tag structure
    "step_count": 0.2,         # Weight for number of reasoning steps
    "verification": 0.2,       # Weight for presence of verification
    "similarity": 0.3          # Weight for similarity to original
}

# Training config
LEARNING_RATE = 5e-5
EPOCHS = 1
WARMUP_RATIO = 0.01
GRADIENT_ACCUMULATION_STEPS = 4
