"""
Global Configuration for Voice-Enabled RAG System
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Auto-load persistent .env file if present
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Dataset Configuration
DATASET_NAME = "ai4bharat/MSMARCO-XI"
DEFAULT_LANGUAGE = "en"  # Default to English (options: 'en', 'hi', 'te', 'bn')
MAX_DATASET_SAMPLES = 500

# Speech-To-Text (STT) Settings
STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_STT_LANG = "en-IN"  # Default English STT language code ('en-IN', 'hi-IN')

# LLM Synthesis (Google Gemini API for Universal Answering & High Accuracy)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_CANDIDATE_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest"
]
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"

def save_env_variable(key: str, value: str):
    """
    Persistently updates an environment variable in os.environ and the .env file.
    """
    os.environ[key] = value
    env_lines = []
    found = False
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k, v = stripped.split("=", 1)
                    if k.strip() == key:
                        env_lines.append(f"{key}={value}\n")
                        found = True
                        continue
                env_lines.append(line if line.endswith("\n") else line + "\n")
    if not found:
        env_lines.append(f"{key}={value}\n")
    
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.writelines(env_lines)


# Vector DB & Embeddings
EMBEDDING_DIM = 256
TOP_K_RETRIEVAL = 3

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
SEMANTIC_SIMILARITY_THRESHOLD = 0.65

# Latency Target SLA (Milliseconds)
TARGET_LATENCY_MS = 200.0

# Guardrail Thresholds
GROUNDING_SIMILARITY_THRESHOLD = 0.35
HALLUCINATION_OVERLAP_THRESHOLD = 0.25

# Offline Demo Mode
ALLOW_SYNTHETIC_DATA_FALLBACK = True
