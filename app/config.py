"""Application configuration."""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Application settings."""
    app_title: str = "Scriptura"
    debug: bool = True
    default_timer_minutes: int = 5

    # AI configuration
    ai_provider: str = field(default_factory=lambda: os.environ.get("AI_PROVIDER", "mock"))
    ai_model: str = field(default_factory=lambda: os.environ.get("AI_MODEL", "gemini-2.5-flash"))
    ai_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    ai_timeout_seconds: int = field(
        default_factory=lambda: int(os.environ.get("AI_TIMEOUT_SECONDS", "30"))
    )


settings = Settings()
