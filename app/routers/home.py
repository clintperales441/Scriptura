"""Home / Dashboard routes."""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def home(request: Request):
    """Render the home/dashboard page."""
    # Phase 2: Replace with real stats from database
    stats = {
        "sessions_completed": 0,
        "current_streak": 0,
        "total_words": 0,
        "recent_sessions": [],
    }
    return templates.TemplateResponse(request, "home.html", {"stats": stats})
