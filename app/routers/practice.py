"""Practice workflow routes."""

import logging

from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.services import prompt_service, practice_service
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
    }


@router.get("/setup")
async def setup(request: Request):
    """Show category and difficulty selection."""
    categories = [
        "Everyday", "Opinion", "Experience",
        "Academic / Professional", "Technical",
    ]
    difficulties = ["Beginner", "Intermediate", "Advanced"]
    timer_options = [3, 5, 10]
    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "categories": categories,
            "difficulties": difficulties,
            "timer_options": timer_options,
            "default_timer": settings.default_timer_minutes,
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
    """Start a new practice session."""
    prompt = prompt_service.get_random_prompt(db, category, difficulty)
    if not prompt:
        return RedirectResponse("/", status_code=303)

    session = practice_service.create_session(
        db,
        prompt_id=prompt.id,
        category=category,
        difficulty=difficulty,
    )

    return templates.TemplateResponse(
        request,
        "practice.html",
        {
            "session": {
                "id": session.id,
                "category": session.category,
                "difficulty": session.difficulty,
                "prompt": {"text": prompt.text},
                "timer_minutes": timer_minutes,
            },
        },
    )


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
    """Submit the rewritten text and show results."""
    session = practice_service.submit_rewrite(db, session_id, rewrite)
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
        {"session": _session_to_template(session)},
    )
