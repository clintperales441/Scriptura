"""Scriptura — English Writing Practice App."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import home, practice

app = FastAPI(title=settings.app_title, debug=settings.debug)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(home.router)
app.include_router(practice.router)
