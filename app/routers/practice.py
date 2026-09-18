"""Practice workflow routes."""

import logging

from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.services import prompt_service, practice_service, progress_service
from app.services.review_service import MockReviewService, ReviewService
from app.schemas.review import ReviewRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/practice")
templates = Jinja2Templates(directory="app/templates")


def _get_review_service() -> ReviewService:
    """Create the appropriate review service based on configuration."""
    if settings.ai_api_key and settings.ai_provider != "mock":
        try:
            from app.services.ai_review_service import AiReviewService
            return AiReviewService(
                api_key=settings.ai_api_key,
                model=settings.ai_model,
                timeout=settings.ai_timeout_seconds,
            )
        except Exception as e:
            logger.warning("Failed to initialize AI review service: %s. Falling back to mock.", e)
            return MockReviewService()
    return MockReviewService()


# Initialize once at module load
_review_service: ReviewService = _get_review_service()


def _session_to_template(session) -> dict:
    """Convert a PracticeSession ORM object to a template-friendly dict."""
    return {
        "id": session.id,
        "category": session.category,
        "difficulty": session.difficulty,
        "prompt": {"text": session.prompt.text},
        "original_text": session.original_text,
        "rewritten_text": session.rewritten_text,
        "original_word_count": session.original_word_count,
        "rewrite_word_count": session.rewrite_word_count,
        "summary": session.summary,
        "overall_feedback": session.overall_feedback,
        "mistakes": [
            {
                "original": m.original_text,
                "correction": m.correction,
                "category": m.category,
                "explanation": m.explanation,
            }
            for m in session.mistakes
        ],
        "original_scores": practice_service.scores_dict(
            session.original_grammar_score,
            session.original_fluency_score,
            session.original_clarity_score,
            session.original_engagement_score,
        ),
        "rewrite_scores": practice_service.scores_dict(
            session.rewrite_grammar_score,
            session.rewrite_fluency_score,
            session.rewrite_clarity_score,
            session.rewrite_engagement_score,
        ),
    }


def _render_practice_start(request, session, prompt_text: str, timer_minutes: int):
    """Shared response builder for a freshly-started session, used by
    both the manual /start route and the weak-area /start-weak-area route
    so the practice.html context stays in one place."""
    return templates.TemplateResponse(
        request,
        "practice.html",
        {
            "session": {
                "id": session.id,
                "category": session.category,
                "difficulty": session.difficulty,
                "prompt": {"text": prompt_text},
                "timer_minutes": timer_minutes,
            },
        },
    )


@router.get("/setup")
async def setup(request: Request, db: Session = Depends(get_db)):
    """Show category and difficulty selection."""
    categories = [
        "Everyday", "Opinion", "Experience",
        "Academic / Professional", "Technical",
    ]
    difficulties = ["Beginner", "Intermediate", "Advanced"]
    timer_options = [3, 5, 10]

    mistake_breakdown = progress_service.get_mistake_breakdown(db)
    weak_area = prompt_service.recommend_topic_category(mistake_breakdown)

    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "categories": categories,
            "difficulties": difficulties,
            "timer_options": timer_options,
            "default_timer": settings.default_timer_minutes,
            # (weak_mistake_category, recommended_topic_category) or None
            "weak_area": weak_area,
        },
    )


@router.post("/start")
async def start(
    request: Request,
    category: str = Form(...),
    difficulty: str = Form(...),
    timer_minutes: int = Form(5),
    db: Session = Depends(get_db),
):
    """Start a new practice session with a manually chosen category."""
    prompt = prompt_service.get_random_prompt(db, category, difficulty)
    if not prompt:
        return RedirectResponse("/", status_code=303)

    session = practice_service.create_session(
        db,
        prompt_id=prompt.id,
        category=category,
        difficulty=difficulty,
    )

    return _render_practice_start(request, session, prompt.text, timer_minutes)


@router.post("/start-weak-area")
async def start_weak_area(
    request: Request,
    difficulty: str = Form(...),
    timer_minutes: int = Form(5),
    db: Session = Depends(get_db),
):
    """Start a new practice session with the topic category chosen for
    the learner, biased toward their current weakest mistake category
    instead of a manual pick."""
    mistake_breakdown = progress_service.get_mistake_breakdown(db)
    prompt, topic_category, _weak_category = prompt_service.get_prompt_for_weak_area(
        db, difficulty, mistake_breakdown
    )

    if not prompt:
        # No mistake history yet (or nothing unresolved) -- there's
        # nothing to recommend from, so don't dead-end the button.
        return RedirectResponse("/practice/setup", status_code=303)

    session = practice_service.create_session(
        db,
        prompt_id=prompt.id,
        category=topic_category,
        difficulty=difficulty,
    )

    return _render_practice_start(request, session, prompt.text, timer_minutes)


@router.post("/submit")
async def submit_writing(
    request: Request,
    session_id: str = Form(...),
    writing: str = Form(...),
    db: Session = Depends(get_db),
):
    """Submit original writing and show review."""
    session = practice_service.get_session(db, session_id)
    if not session:
        return RedirectResponse("/", status_code=303)

    # Get review from the review service
    review_request = ReviewRequest(
        prompt=session.prompt.text,
        writing=writing,
        category=session.category,
        difficulty=session.difficulty,
    )
    review_response = _review_service.review(review_request)

    # Save writing and review to database
    session = practice_service.submit_writing(db, session_id, writing, review_response)
    if not session:
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request,
        "review.html",
        {"session": _session_to_template(session)},
    )


@router.get("/rewrite/{session_id}")
async def rewrite_form(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db),
):
    """Show the rewrite form."""
    session = practice_service.get_session(db, session_id)
    if not session:
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request,
        "rewrite.html",
        {"session": _session_to_template(session)},
    )


@router.post("/rewrite/{session_id}")
async def submit_rewrite(
    request: Request,
    session_id: str,
    rewrite: str = Form(...),
    db: Session = Depends(get_db),
):
    """Submit the rewritten text, re-score it, and show results."""
    existing_session = practice_service.get_session(db, session_id)
    if not existing_session:
        return RedirectResponse("/", status_code=303)

    # Re-score the rewrite (a second review call) so the result page can
    # show a before/after comparison, not just the original's scores.
    # We deliberately ignore this second call's mistakes/summary -- only
    # its `scores` are used, so we don't double up WritingMistake rows
    # for what's fundamentally the same session's mistake history.
    rewrite_review_request = ReviewRequest(
        prompt=existing_session.prompt.text,
        writing=rewrite,
        category=existing_session.category,
        difficulty=existing_session.difficulty,
    )
    rewrite_review = _review_service.review(rewrite_review_request)

    session = practice_service.submit_rewrite(db, session_id, rewrite, rewrite_review)
    if not session:
        return RedirectResponse("/", status_code=303)

    return RedirectResponse(f"/practice/result/{session_id}", status_code=303)


@router.get("/result/{session_id}")
async def result(
    request: Request,
    session_id: str,
    db: Session = Depends(get_db),
):
    """Show session results."""
    session = practice_service.get_session(db, session_id)
    if not session:
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "session": _session_to_template(session),
            "corrected_text": practice_service.build_corrected_text(session),
        },
    )
