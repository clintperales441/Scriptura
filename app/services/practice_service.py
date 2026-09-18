"""Practice service — create and manage practice sessions."""

import re
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

    if review.scores:
        session.original_grammar_score = review.scores.grammar
        session.original_fluency_score = review.scores.fluency
        session.original_clarity_score = review.scores.clarity
        session.original_engagement_score = review.scores.engagement

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


def _flagged_text_still_present(original_text: str, rewrite: str) -> bool:
    """Check whether the mistake's flagged text still appears in the
    rewrite, as a whole word/phrase and case-sensitively.

    Word boundaries matter: without them, a single flagged letter like "i"
    (capitalization mistakes) would match inside ordinary words such as
    "like" or "it", making every rewrite look uncorrected. Case matters
    too, since some corrections (like "i" -> "I") are capitalization-only
    -- a case-insensitive check could never detect those as fixed.
    """
    pattern = r"\b" + re.escape(original_text.strip()) + r"\b"
    return re.search(pattern, rewrite) is not None


def _mark_corrected_mistakes(session: PracticeSession, rewrite: str) -> int:
    """Mark each mistake as corrected if its flagged text no longer
    appears in the rewrite.

    This is a text-presence check, not a semantic diff -- it will miss
    cases where the learner rephrased a sentence entirely without keeping
    the flagged snippet's exact wording, and it can be fooled if the
    flagged phrase coincidentally reappears elsewhere for an unrelated
    reason. It's a reasonable v1 signal; a stricter check (e.g. asking the
    AI to re-verify) can replace this later without changing the
    surrounding call sites.

    Returns the number of mistakes marked as corrected.
    """
    corrected_count = 0

    for mistake in session.mistakes:
        was_corrected = not _flagged_text_still_present(mistake.original_text, rewrite)
        mistake.corrected_by_rewrite = was_corrected
        if was_corrected:
            corrected_count += 1

    return corrected_count


def submit_rewrite(
    db: Session,
    session_id: str,
    rewrite: str,
    rewrite_review: ReviewResponse | None = None,
) -> PracticeSession | None:
    """Save the rewritten text, mark which mistakes were corrected, store
    the rewrite's own quality scores if provided, and mark the session
    complete.

    `rewrite_review` is optional (defaults to None) so existing callers
    that don't re-score the rewrite still work -- the session just won't
    have rewrite_* scores populated, and any before/after UI should
    handle that as "not yet scored" rather than assuming it's always set.
    """
    session = get_session(db, session_id)
    if not session:
        return None

    session.rewritten_text = rewrite
    session.rewrite_word_count = count_words(rewrite)
    session.corrected_mistake_count = _mark_corrected_mistakes(session, rewrite)

    if rewrite_review and rewrite_review.scores:
        session.rewrite_grammar_score = rewrite_review.scores.grammar
        session.rewrite_fluency_score = rewrite_review.scores.fluency
        session.rewrite_clarity_score = rewrite_review.scores.clarity
        session.rewrite_engagement_score = rewrite_review.scores.engagement

    session.completed = True
    session.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session


def scores_dict(
    grammar: int | None,
    fluency: int | None,
    clarity: int | None,
    engagement: int | None,
) -> dict | None:
    """Build a template-friendly scores dict (with a computed `overall`)
    from four raw score columns, or None if they haven't been scored yet.

    Shared by review/result page rendering so the "not yet scored"
    check and overall-average math live in exactly one place.
    """
    if grammar is None:
        return None

    return {
        "grammar": grammar,
        "fluency": fluency,
        "clarity": clarity,
        "engagement": engagement,
        "overall": round((grammar + fluency + clarity + engagement) / 4),
    }


def build_corrected_text(session: PracticeSession) -> str:
    """Assemble a single fully-corrected version of the original text by
    applying each flagged mistake's suggested correction in place.

    This is a snippet-level find-and-replace, not a fresh AI rewrite of
    the whole passage -- each mistake's flagged text is swapped for its
    suggested correction (first occurrence only, to avoid clobbering an
    unrelated identical snippet elsewhere in the text). If a mistake's
    flagged text can't be found verbatim (e.g. it was itself a whole-
    sentence restructuring suggestion), that one replacement is skipped
    and the rest still apply. It's meant as a reference to compare
    against the learner's own rewrite, not as a polished model answer.
    """
    corrected = session.original_text or ""

    for mistake in session.mistakes:
        corrected = corrected.replace(mistake.original_text, mistake.correction, 1)

    return corrected
