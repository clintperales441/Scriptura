"""Progress service — calculate dashboard statistics."""

from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.models.practice_session import PracticeSession


def get_stats(db: Session) -> dict:
    """Get dashboard statistics."""
    # Total completed sessions
    sessions_completed = db.scalar(
        select(func.count(PracticeSession.id)).where(
            PracticeSession.completed == True  # noqa: E712
        )
    ) or 0

    # Total words written (original + rewrite)
    total_original = db.scalar(
        select(func.sum(PracticeSession.original_word_count)).where(
            PracticeSession.completed == True  # noqa: E712
        )
    ) or 0
    total_rewrite = db.scalar(
        select(func.sum(PracticeSession.rewrite_word_count)).where(
            PracticeSession.completed == True  # noqa: E712
        )
    ) or 0
    total_words = total_original + total_rewrite

    # Current streak (consecutive days with at least one completed session)
    current_streak = _calculate_streak(db)

    # Recent sessions (last 10)
    recent_stmt = (
        select(PracticeSession)
        .where(PracticeSession.completed == True)  # noqa: E712
        .order_by(PracticeSession.completed_at.desc())
        .limit(10)
    )
    recent_sessions = db.scalars(recent_stmt).unique().all()

    return {
        "sessions_completed": sessions_completed,
        "current_streak": current_streak,
        "total_words": total_words,
        "recent_sessions": [
            {
                "id": s.id,
                "category": s.category,
                "difficulty": s.difficulty,
                "date": s.completed_at.strftime("%b %d, %Y") if s.completed_at else "",
                "word_count": s.original_word_count,
                "mistake_count": s.mistake_count,
            }
            for s in recent_sessions
        ],
    }


def _calculate_streak(db: Session) -> int:
    """Calculate consecutive days with completed sessions."""
    # Get distinct dates of completed sessions, ordered descending
    stmt = (
        select(func.date(PracticeSession.completed_at))
        .where(PracticeSession.completed == True)  # noqa: E712
        .group_by(func.date(PracticeSession.completed_at))
        .order_by(func.date(PracticeSession.completed_at).desc())
    )
    dates = db.execute(stmt).scalars().all()

    if not dates:
        return 0

    today = datetime.now(timezone.utc).date()
    streak = 0

    for i, date_str in enumerate(dates):
        # SQLite returns date as string 'YYYY-MM-DD'
        if isinstance(date_str, str):
            session_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            session_date = date_str

        expected_date = today - timedelta(days=i)
        if session_date == expected_date:
            streak += 1
        elif i == 0 and session_date == today - timedelta(days=1):
            # Allow streak to count from yesterday if no session today yet
            streak += 1
            today = today - timedelta(days=1)
        else:
            break

    return streak
