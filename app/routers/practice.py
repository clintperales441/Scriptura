"""Practice workflow routes."""

import json
import random
import uuid
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings

router = APIRouter(prefix="/practice")
templates = Jinja2Templates(directory="app/templates")

# In-memory session store (replaced by database in Phase 2)
_sessions: dict[str, dict] = {}

# Load prompts from JSON
_prompts_path = Path(__file__).parent.parent / "data" / "prompts.json"
with open(_prompts_path, "r", encoding="utf-8") as f:
    _all_prompts = json.load(f)


def _pick_prompt(category: str, difficulty: str) -> dict:
    """Pick a random prompt matching category and difficulty."""
    matching = [
        p for p in _all_prompts
        if p["category"] == category and p["difficulty"] == difficulty
    ]
    if not matching:
        matching = [p for p in _all_prompts if p["category"] == category]
    if not matching:
        matching = _all_prompts
    return random.choice(matching)


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split()) if text and text.strip() else 0


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
):
    """Start a new practice session."""
    prompt = _pick_prompt(category, difficulty)
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "difficulty": difficulty,
        "prompt": prompt,
        "timer_minutes": timer_minutes,
        "original_text": None,
        "rewritten_text": None,
        "mistakes": [],
        "completed": False,
    }
    return templates.TemplateResponse(
        request,
        "practice.html",
        {"session": _sessions[session_id]},
    )


@router.post("/submit")
async def submit_writing(
    request: Request,
    session_id: str = Form(...),
    writing: str = Form(...),
):
    """Submit original writing and show review."""
    session = _sessions.get(session_id)
    if not session:
        return RedirectResponse("/", status_code=303)

    session["original_text"] = writing
    session["original_word_count"] = _count_words(writing)

    # Phase 3: Replace with real AI review
    session["mistakes"] = [
        {
            "original": "(AI review not yet connected)",
            "correction": "(will appear here)",
            "category": "Info",
            "explanation": (
                "AI-powered writing review will be integrated in Phase 3. "
                "For now, proceed to the rewrite step to practice revising "
                "your writing on your own."
            ),
        }
    ]
    session["summary"] = (
        "AI review will be available soon. For now, re-read your writing "
        "and try to improve it yourself."
    )
    session["overall_feedback"] = ""

    return templates.TemplateResponse(
        request,
        "review.html",
        {"session": session},
    )


@router.get("/rewrite/{session_id}")
async def rewrite_form(request: Request, session_id: str):
    """Show the rewrite form."""
    session = _sessions.get(session_id)
    if not session:
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request,
        "rewrite.html",
        {"session": session},
    )


@router.post("/rewrite/{session_id}")
async def submit_rewrite(
    request: Request,
    session_id: str,
    rewrite: str = Form(...),
):
    """Submit the rewritten text and show results."""
    session = _sessions.get(session_id)
    if not session:
        return RedirectResponse("/", status_code=303)

    session["rewritten_text"] = rewrite
    session["rewrite_word_count"] = _count_words(rewrite)
    session["completed"] = True
    session["completed_at"] = datetime.now(timezone.utc).isoformat()

    return RedirectResponse(f"/practice/result/{session_id}", status_code=303)


@router.get("/result/{session_id}")
async def result(request: Request, session_id: str):
    """Show session results."""
    session = _sessions.get(session_id)
    if not session:
        return RedirectResponse("/", status_code=303)

    return templates.TemplateResponse(
        request,
        "result.html",
        {"session": session},
    )
