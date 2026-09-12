"""Scriptura — English Writing Practice App."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import engine, SessionLocal
from app.models import Base
from app.seed import seed_prompts
from app.routers import home, practice

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables and seed prompts on startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created.")

    db = SessionLocal()
    try:
        inserted = seed_prompts(db)
        if inserted:
            logger.info("Seeded %d prompts.", inserted)
    finally:
        db.close()

    yield


app = FastAPI(title=settings.app_title, debug=settings.debug, lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(home.router)
app.include_router(practice.router)
