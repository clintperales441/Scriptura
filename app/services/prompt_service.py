"""Prompt service — query and select writing prompts."""

import random

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.writing_prompt import WritingPrompt


def get_random_prompt(
    db: Session,
    category: str,
    difficulty: str,
) -> WritingPrompt | None:
    """Pick a random active prompt matching category and difficulty."""
    stmt = (
        select(WritingPrompt)
        .where(
            WritingPrompt.active == True,  # noqa: E712
            WritingPrompt.category == category,
            WritingPrompt.difficulty == difficulty,
        )
    )
    prompts = db.scalars(stmt).all()

    if not prompts:
        # Fallback: match category only
        stmt = select(WritingPrompt).where(
            WritingPrompt.active == True,  # noqa: E712
            WritingPrompt.category == category,
        )
        prompts = db.scalars(stmt).all()

    if not prompts:
        # Fallback: any active prompt
        stmt = select(WritingPrompt).where(WritingPrompt.active == True)  # noqa: E712
        prompts = db.scalars(stmt).all()

    return random.choice(prompts) if prompts else None
