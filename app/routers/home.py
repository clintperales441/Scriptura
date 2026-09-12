"""Home / Dashboard routes."""

from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.services import progress_service

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def home(request: Request, db: Session = Depends(get_db)):
    """Render the home/dashboard page."""
    stats = progress_service.get_stats(db)
    return templates.TemplateResponse(request, "home.html", {"stats": stats})
