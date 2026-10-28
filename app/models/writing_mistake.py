"""WritingMistake model."""

import json

from sqlalchemy import String, Text, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class WritingMistake(Base):
    """A single mistake identified during review."""

    __tablename__ = "writing_mistakes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("practice_sessions.id"), nullable=False, index=True
    )
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    correction: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_by_rewrite: Mapped[bool] = mapped_column(Boolean, default=False)
    rule_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    source_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    proof: Mapped[str | None] = mapped_column(Text, nullable=True)
    breakdown_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationship
    session: Mapped["PracticeSession"] = relationship(back_populates="mistakes")

    @property
    def breakdown(self) -> list[dict]:
        """Deserialize structured word/POS/punctuation breakdown steps."""
        if not self.breakdown_json:
            return []
        try:
            parsed = json.loads(self.breakdown_json)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []


from app.models.practice_session import PracticeSession  # noqa: E402
