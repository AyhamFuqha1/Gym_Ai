import os
from dotenv import load_dotenv

load_dotenv()

def _bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

def _list_env(name: str, default: str = "*") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]

class Settings:
    def __init__(self):
        # Server configuration
        self.APP_NAME = os.getenv("APP_NAME", "Gym AI Service")
        self.DEBUG = _bool_env("DEBUG", False)
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = _int_env("PORT", 8080)
        self.CORS_ALLOW_ORIGINS = _list_env("CORS_ALLOW_ORIGINS", "*")

        # MySQL configuration
        self.DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
        self.DB_PORT = _int_env("DB_PORT", 3306)
        self.DB_USERNAME = os.getenv("DB_USERNAME", "root")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "")
        self.DB_DATABASE = os.getenv("DB_DATABASE", "gym")

        # Embedding/LLM provider configuration. OpenRouter is configured for
        # future phases only; Phase 1 does not call it.
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.GEMINI_EMBEDDING_MODEL = os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "models/gemini-embedding-2-preview",
        )
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")

        # Chroma/vector store configuration
        self.CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_gym")
        self.EXERCISES_COLLECTION = os.getenv("EXERCISES_COLLECTION", "exercises_data")
        self.NUTRITION_COLLECTION = os.getenv("NUTRITION_COLLECTION", "nutrition_data")

        # Sync files and batching
        self.EXERCISES_MANIFEST = os.getenv("EXERCISES_MANIFEST", "gym_manifest.json")
        self.NUTRITION_MANIFEST = os.getenv("NUTRITION_MANIFEST", "nutrition_manifest.json")
        self.CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "./data/sync_checkpoint.json")
        self.BATCH_SIZE = _int_env("BATCH_SIZE", 50)
        self.SYNC_DELAY_SECONDS = _float_env("SYNC_DELAY_SECONDS", 1.0)

        # Optional internal auth token for future enforcement. Do not enforce
        # yet, to keep current Laravel calls backward compatible.
        self.INTERNAL_AI_API_TOKEN = os.getenv("INTERNAL_AI_API_TOKEN", "")

    @property
    def DB_CONFIG(self):
        return {
            "host": self.DB_HOST,
            "port": self.DB_PORT,
            "user": self.DB_USERNAME,
            "password": self.DB_PASSWORD,
            "database": self.DB_DATABASE,
        }

settings = Settings()
