from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment.

    Uses pydantic-settings so values can come from environment variables
    (e.g. GEMINI_API_KEY).
    """

    gemini_api_key: Optional[str] = None

    class Config:
        env_prefix = ""  # read GEMINI_API_KEY as-is


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
