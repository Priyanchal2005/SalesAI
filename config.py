"""Application configuration for SalesPilot AI."""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

APP_NAME = "SalesPilot AI"
APP_TAGLINE = "Autonomous AI Sales Voice & Revenue Qualification Agent"
COMPANY_NAME = "Fortivex Solutions"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

DATABASE_PATH = str(BASE_DIR / os.getenv("DATABASE_PATH", "salespilot.db"))
DATA_DIR = BASE_DIR / "data"
SAMPLE_KNOWLEDGE_PATH = DATA_DIR / "sample_knowledge.txt"

# RAG settings
CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
TOP_K_CHUNKS = 3

# Supported languages
SUPPORTED_LANGUAGES = {
    "English": {"code": "en", "sr_locale": "en-IN", "tts_lang": "en"},
    "Hindi": {"code": "hi", "sr_locale": "hi-IN", "tts_lang": "hi"},
}

# Appointment slot configuration (internal prototype slots)
AVAILABLE_TIME_SLOTS = [
    "10:00 AM",
    "11:30 AM",
    "02:00 PM",
    "03:30 PM",
    "05:00 PM",
]
