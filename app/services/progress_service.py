"""Progress service — calculate dashboard statistics."""

from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import select, func, Integer

from app.models.practice_session import PracticeSession
from app.models.writing_mistake import WritingMistake

# How many of the learner's most recent completed sessions to consider for
# the mistake breakdown. Bounded (rather than all-time) so the signal
# reflects current ability, not a diluted lifetime average.
MISTAKE_HISTORY_SESSION_WINDOW = 20

# How many recent scored sessions to plot on the score trend chart.
SCORE_TREND_SESSION_WINDOW = 10

_SCORE_DIMENSIONS = ["grammar", "fluency", "clarity", "engagement", "overall"]
_CHART_WIDTH = 600
_CHART_HEIGHT = 160
_CHART_PADDING_X = 16
# Vertical space the plotted line uses within the chart height, leaving a
# small margin top and bottom so a 100 or 0 score isn't clipped at the edge.
_CHART_PLOT_HEIGHT = 140
_CHART_TOP_MARGIN = 10


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


def get_mistake_breakdown(
    db: Session, session_window: int = MISTAKE_HISTORY_SESSION_WINDOW
) -> list[dict]:
    """Group mistakes by category across the learner's most recent
    completed sessions, with counts and correction rate.

    This is what turns individual review results into an actual learning
    signal: a category with a high count and a low correction rate is a
    recurring weakness the learner hasn't been fixing on rewrite.
    """
    # Most recent N completed session IDs, used to bound the window.
    recent_session_ids_stmt = (
        select(PracticeSession.id)
        .where(PracticeSession.completed == True)  # noqa: E712
        .order_by(PracticeSession.completed_at.desc())
        .limit(session_window)
    )
    recent_session_ids = db.scalars(recent_session_ids_stmt).all()

    if not recent_session_ids:
        return []

    breakdown_stmt = (
        select(
            WritingMistake.category,
            func.count(WritingMistake.id).label("total"),
            func.sum(
                func.cast(WritingMistake.corrected_by_rewrite, Integer)
            ).label("corrected"),
        )
        .where(WritingMistake.session_id.in_(recent_session_ids))
        .group_by(WritingMistake.category)
        .order_by(func.count(WritingMistake.id).desc())
    )
    rows = db.execute(breakdown_stmt).all()

    breakdown = []
    for category, total, corrected in rows:
        corrected = corrected or 0
        breakdown.append(
            {
                "category": category,
                "total": total,
                "corrected": corrected,
                "correction_rate": round((corrected / total) * 100) if total else 0,
            }
        )

    return breakdown


def get_score_trend(
    db: Session, session_window: int = SCORE_TREND_SESSION_WINDOW
) -> list[dict]:
    """Return the learner's original-writing scores for their most
    recent completed, scored sessions, oldest first (so a chart reads
    left-to-right as "earlier -> now").

    Only the *original* submission's scores are used, not the rewrite's --
    this is meant to track how the learner's unaided first attempts are
    trending over time, not how much any single rewrite happened to help.
    Sessions from before scoring existed (or where the AI review failed)
    have no original_grammar_score and are skipped.
    """
    stmt = (
        select(PracticeSession)
        .where(
            PracticeSession.completed == True,  # noqa: E712
            PracticeSession.original_grammar_score.is_not(None),
        )
        .order_by(PracticeSession.completed_at.desc())
        .limit(session_window)
    )
    sessions = list(reversed(db.scalars(stmt).unique().all()))

    trend = []
    for s in sessions:
        overall = round(
            (
                s.original_grammar_score
                + s.original_fluency_score
                + s.original_clarity_score
                + s.original_engagement_score
            )
            / 4
        )
        trend.append(
            {
                "completed_at": s.completed_at,
                "grammar": s.original_grammar_score,
                "fluency": s.original_fluency_score,
                "clarity": s.original_clarity_score,
                "engagement": s.original_engagement_score,
                "overall": overall,
            }
        )

    return trend


def get_score_trend_chart(
    db: Session, session_window: int = SCORE_TREND_SESSION_WINDOW
) -> dict | None:
    """Build ready-to-render SVG polyline points for a score trend chart,
    so the template only has to draw <polyline> elements rather than do
    coordinate math in Jinja.

    Returns None if there are fewer than 2 scored sessions -- a single
    point can't show a trend.
    """
    trend = get_score_trend(db, session_window)
    if len(trend) < 2:
        return None

    n = len(trend)
    step = (_CHART_WIDTH - 2 * _CHART_PADDING_X) / (n - 1)

    def x_for(i: int) -> float:
        return _CHART_PADDING_X + i * step

    def y_for(score: int) -> float:
        # score=100 -> near the top, score=0 -> near the bottom.
        return _CHART_TOP_MARGIN + _CHART_PLOT_HEIGHT - (score / 100) * _CHART_PLOT_HEIGHT

    lines = {
        dim: " ".join(
            f"{x_for(i):.1f},{y_for(point[dim]):.1f}" for i, point in enumerate(trend)
        )
        for dim in _SCORE_DIMENSIONS
    }

    return {
        "width": _CHART_WIDTH,
        "height": _CHART_HEIGHT,
        "lines": lines,
        "latest": trend[-1],
        "session_count": n,
    }
