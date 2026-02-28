from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError

# Define the file path
env_path = Path(".env")

if not env_path.exists():
    raise FileNotFoundError(
        f"Critical error: Configuration file '{env_path}' not found. "
        "Please create one based on .env.example"
    )


class Settings(BaseSettings):
    """
    Docstring for Settings
    """
    ALLOWED_ORIGINS: str
    SECRET_API_KEY: str
    DATABASE_URL: str
    OLLAMA_BASE_URL: str
    CHROMA_PATH: str
    MODEL_NAME: str
    SIMILARITY_THRESHOLD: float
    PRODUCTION: bool
    EMBEDDING_MODEL: str
    GROQ_API_KEY: str

    # Use SettingsConfigDict for better structure
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",         # This ignores extra variables like 'production'
        case_sensitive=False    # Usually best for env vars
    )


try:
    settings = Settings()
    print("Settings loaded successfully!")
except ValidationError as e:
    print(f"Configuration validation error:\n{e}")
