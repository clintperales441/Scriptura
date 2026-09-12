"""Application configuration."""

from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    app_title: str = "Scriptura"
    debug: bool = True
    default_timer_minutes: int = 5


settings = Settings()
