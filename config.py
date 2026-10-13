"""All settings in one place. Override any of them with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Ollama models. qwen2.5:7b handles Arabic far better than llama3:8b;
# bge-m3 is a multilingual embedding model, so an Arabic question can
# retrieve an English passage and vice versa.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Where your PDFs live. The r"..." (raw string) keeps Windows backslashes as-is,
# and Path handles the spaces in the folder names.
DEFAULT_DATA_DIR = r"C:\Users\RAWAF\OneDrive - University of Jeddah\Desktop\sdaia AI files"
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
DB_DIR = Path(os.getenv("DB_DIR", BASE_DIR / "chroma_db"))
COLLECTION = "sdaia_ai"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
