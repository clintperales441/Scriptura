"""WritingMistake model."""

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

    # Relationship
    session: Mapped["PracticeSession"] = relationship(back_populates="mistakes")


from app.models.practice_session import PracticeSession  # noqa: E402
