from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Cap per-chunk LLM knowledge extraction during upload. A large PDF can produce
# hundreds of chunks; calling Groq once per chunk makes uploads appear stuck.
MAX_KNOWLEDGE_CHUNKS = int(os.getenv("MAX_KNOWLEDGE_CHUNKS", "25"))
KNOWLEDGE_EXTRACTION_ENABLED = (
    os.getenv("KNOWLEDGE_EXTRACTION_ENABLED", "true").lower() == "true"
)

# Docling downloads large ML models and can take 20+ minutes on first PDF in Docker.
# Default to fast poppler-based extraction; set USE_DOCLING=true for layout-aware parsing.
USE_DOCLING = os.getenv("USE_DOCLING", "false").lower() == "true"
DOCLING_TIMEOUT_SECONDS = int(os.getenv("DOCLING_TIMEOUT_SECONDS", "120"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "industrial_chunks")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "industrialmind")

UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "data" / "uploads"))
PROCESSED_FOLDER = Path(os.getenv("PROCESSED_FOLDER", BASE_DIR / "data" / "processed"))
STORAGE_FOLDER = Path(os.getenv("STORAGE_FOLDER", BASE_DIR / "data" / "storage"))

for folder in (UPLOAD_FOLDER, PROCESSED_FOLDER, STORAGE_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)
