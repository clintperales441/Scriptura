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

    # Ensure any existing writing_mistakes table has new RAG columns
    with engine.begin() as conn:
        cursor = conn.exec_driver_sql("PRAGMA table_info(writing_mistakes)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if existing_cols:
            if "rule_name" not in existing_cols:
                conn.exec_driver_sql("ALTER TABLE writing_mistakes ADD COLUMN rule_name VARCHAR(150)")
            if "source_name" not in existing_cols:
                conn.exec_driver_sql("ALTER TABLE writing_mistakes ADD COLUMN source_name VARCHAR(100)")
            if "source_url" not in existing_cols:
                conn.exec_driver_sql("ALTER TABLE writing_mistakes ADD COLUMN source_url TEXT")
            if "proof" not in existing_cols:
                conn.exec_driver_sql("ALTER TABLE writing_mistakes ADD COLUMN proof TEXT")
            if "breakdown_json" not in existing_cols:
                conn.exec_driver_sql("ALTER TABLE writing_mistakes ADD COLUMN breakdown_json TEXT")

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
