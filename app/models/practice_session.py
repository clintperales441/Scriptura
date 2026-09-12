"""PracticeSession model."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Text, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class PracticeSession(Base):
    """A single practice session from prompt through rewrite."""

    __tablename__ = "practice_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    prompt_id: Mapped[int] = mapped_column(
        ForeignKey("writing_prompts.id"), nullable=False
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False)
    original_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rewritten_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_word_count: Mapped[int] = mapped_column(Integer, default=0)
    rewrite_word_count: Mapped[int] = mapped_column(Integer, default=0)
    mistake_count: Mapped[int] = mapped_column(Integer, default=0)
    corrected_mistake_count: Mapped[int] = mapped_column(Integer, default=0)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    overall_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    prompt: Mapped["WritingPrompt"] = relationship(lazy="joined")
    mistakes: Mapped[list["WritingMistake"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", lazy="joined"
    )


# Avoid circular import
from app.models.writing_prompt import WritingPrompt  # noqa: E402
from app.models.writing_mistake import WritingMistake  # noqa: E402
