import html
import re
import uuid
from datetime import datetime, timezone

from markupsafe import Markup
from sqlalchemy.orm import Session

from app.models.practice_session import PracticeSession
from app.models.writing_mistake import WritingMistake
from app.schemas.review import ReviewResponse, WritingScores


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
            rule_name=mistake_data.rule_name,
            source_name=mistake_data.source_name,
            source_url=mistake_data.source_url,
            proof=mistake_data.proof,
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
    "like" or "it", making every rewrite look uncorrected. Boundary checks
    are applied at word transitions, while pure punctuation tokens are
    matched without breaking on non-word boundaries.
    """
    text = (original_text or "").strip()
    if not text:
        return False

    prefix = r"\b" if re.match(r"^\w", text) else ""
    suffix = r"\b" if re.search(r"\w$", text) else ""
    pattern = prefix + re.escape(text) + suffix
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
    rewrite_review: ReviewResponse | WritingScores | None = None,
) -> PracticeSession | None:
    """Save the rewritten text, mark which mistakes were corrected, store
    the rewrite's own quality scores if provided, and mark the session
    complete.

    `rewrite_review` can be either a WritingScores instance (from lightweight
    score-only review) or a full ReviewResponse, or None.
    """
    session = get_session(db, session_id)
    if not session:
        return None

    session.rewritten_text = rewrite
    session.rewrite_word_count = count_words(rewrite)
    session.corrected_mistake_count = _mark_corrected_mistakes(session, rewrite)

    scores = rewrite_review if isinstance(rewrite_review, WritingScores) else (
        rewrite_review.scores if rewrite_review else None
    )

    if scores:
        session.rewrite_grammar_score = scores.grammar
        session.rewrite_fluency_score = scores.fluency
        session.rewrite_clarity_score = scores.clarity
        session.rewrite_engagement_score = scores.engagement

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

    Applies word-boundary anchoring when matching word tokens so that short
    flagged letters (like "i") or words (like "to") do not replace characters
    inside unrelated words (e.g. "This" -> "ThIs" or "today" -> "tooday").
    Instructional placeholders (like "(delete)" or "(remove)") are cleanly
    treated as deletions.
    """
    corrected = session.original_text or ""

    for mistake in session.mistakes:
        orig = (mistake.original_text or "").strip()
        if not orig:
            continue

        correction = mistake.correction or ""
        # Handle instructional deletion placeholders
        if correction.strip().lower() in ("(delete)", "(remove)", "(none)", "(omit)", "(delete comma)"):
            replacement = ""
        else:
            replacement = correction

        prefix = r"\b" if re.match(r"^\w", orig) else ""
        suffix = r"\b" if re.search(r"\w$", orig) else ""
        pattern = prefix + re.escape(orig) + suffix

        if re.search(pattern, corrected):
            corrected = re.sub(pattern, lambda _: replacement, corrected, count=1)
        elif orig in corrected:
            corrected = corrected.replace(orig, replacement, 1)

    return corrected


def render_highlighted_text(text: str, mistakes: list) -> Markup:
    """Render original writing as safe HTML with AI-recommended words highlighted.

    Each highlighted word is wrapped in an interactive element with data attributes
    for the word correction, category, authoritative proof, and Merriam-Webster link,
    enabling the instant hover proof box.
    """
    if not text:
        return Markup("")

    if not mistakes:
        return Markup(html.escape(text))

    def _get_val(obj, key, default=""):
        if isinstance(obj, dict):
            return obj.get(key, default) or default
        return getattr(obj, key, default) or default

    # Collect matched intervals: (start, end, index, mistake_dict)
    claimed_spans: list[tuple[int, int]] = []
    matches: list[tuple[int, int, int, dict]] = []

    for idx, mistake in enumerate(mistakes):
        orig = _get_val(mistake, "original_text") or _get_val(mistake, "original")
        orig = (orig or "").strip()
        if not orig:
            continue

        prefix = r"\b" if re.match(r"^\w", orig) else ""
        suffix = r"\b" if re.search(r"\w$", orig) else ""
        pattern = prefix + re.escape(orig) + suffix

        # Try word-boundary regex first, then exact substring
        found_span = None
        for m in re.finditer(pattern, text):
            s, e = m.span()
            if not any(cs <= s < ce or cs < e <= ce for cs, ce in claimed_spans):
                found_span = (s, e)
                break

        if not found_span:
            # Fallback substring search
            start_pos = 0
            while True:
                pos = text.find(orig, start_pos)
                if pos == -1:
                    break
                s, e = pos, pos + len(orig)
                if not any(cs <= s < ce or cs < e <= ce for cs, ce in claimed_spans):
                    found_span = (s, e)
                    break
                start_pos = pos + 1

        if found_span:
            claimed_spans.append(found_span)
            m_dict = {
                "original": orig,
                "correction": _get_val(mistake, "correction"),
                "category": _get_val(mistake, "category"),
                "explanation": _get_val(mistake, "explanation"),
                "rule_name": _get_val(mistake, "rule_name") or "Standard English Usage",
                "source_name": _get_val(mistake, "source_name") or "Merriam-Webster",
                "source_url": _get_val(mistake, "source_url") or f"https://www.merriam-webster.com/dictionary/{orig.lower()}",
                "proof": _get_val(mistake, "proof") or _get_val(mistake, "explanation"),
            }
            matches.append((found_span[0], found_span[1], idx, m_dict))

    # Sort matches chronologically by text index
    matches.sort(key=lambda x: x[0])

    html_parts: list[str] = []
    cursor = 0

    for s, e, idx, m_info in matches:
        if s > cursor:
            html_parts.append(html.escape(text[cursor:s]))

        matched_text = text[s:e]
        esc_orig = html.escape(str(m_info["original"]))
        esc_corr = html.escape(str(m_info["correction"]))
        esc_cat = html.escape(str(m_info["category"]))
        esc_rule = html.escape(str(m_info["rule_name"]))
        esc_src = html.escape(str(m_info["source_name"]))
        esc_url = html.escape(str(m_info["source_url"]))
        esc_proof = html.escape(str(m_info["proof"]))
        esc_exp = html.escape(str(m_info["explanation"]))
        esc_word = html.escape(matched_text)

        html_parts.append(
            f'<mark class="ai-highlight-word" '
            f'data-index="{idx}" '
            f'data-original="{esc_orig}" '
            f'data-correction="{esc_corr}" '
            f'data-category="{esc_cat}" '
            f'data-rule-name="{esc_rule}" '
            f'data-source-name="{esc_src}" '
            f'data-source-url="{esc_url}" '
            f'data-proof="{esc_proof}" '
            f'data-explanation="{esc_exp}" '
            f'tabindex="0">'
            f'{esc_word}'
            f'</mark>'
        )
        cursor = e

    if cursor < len(text):
        html_parts.append(html.escape(text[cursor:]))

    return Markup("".join(html_parts))
