"""Practice service — create and manage practice sessions."""

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.practice_session import PracticeSession
from app.models.writing_mistake import WritingMistake
from app.schemas.review import ReviewResponse


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split()) if text and text.strip() else 0


def create_session(
    db: Session,
    prompt_id: int,
    category: str,
    difficulty: str,
) -> PracticeSession:
    """Create a new practice session."""
    session = PracticeSession(
        id=str(uuid.uuid4()),
        started_at=datetime.now(timezone.utc),
        prompt_id=prompt_id,
        category=category,
        difficulty=difficulty,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: str) -> PracticeSession | None:
    """Get a practice session by ID."""
    return db.get(PracticeSession, session_id)


def submit_writing(
    db: Session,
    session_id: str,
    writing: str,
    review: ReviewResponse,
) -> PracticeSession | None:
    """Save the original writing and review results."""
    session = get_session(db, session_id)
    if not session:
        return None

    session.original_text = writing
    session.original_word_count = count_words(writing)
    session.summary = review.summary
    session.overall_feedback = review.overall_feedback

    # Save mistakes
    for mistake_data in review.mistakes:
        mistake = WritingMistake(
            session_id=session_id,
            original_text=mistake_data.original,
            correction=mistake_data.correction,
            category=mistake_data.category,
            explanation=mistake_data.explanation,
        )
        db.add(mistake)

    session.mistake_count = len(review.mistakes)
    db.commit()
    db.refresh(session)
    return session


def submit_rewrite(
    db: Session,
    session_id: str,
    rewrite: str,
) -> PracticeSession | None:
    """Save the rewritten text and mark session complete."""
    session = get_session(db, session_id)
    if not session:
        return None

    session.rewritten_text = rewrite
    session.rewrite_word_count = count_words(rewrite)
    session.completed = True
    session.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session
