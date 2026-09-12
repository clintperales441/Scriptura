"""Seed prompts into the database on first startup."""

import json
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.models.writing_prompt import WritingPrompt


def seed_prompts(db: Session) -> int:
    """Load prompts from JSON into the database if table is empty.

    Returns the number of prompts inserted.
    """
    count = db.scalar(select(func.count(WritingPrompt.id)))
    if count and count > 0:
        return 0  # Already seeded

    prompts_path = Path(__file__).parent / "data" / "prompts.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    inserted = 0
    for p in prompts_data:
        prompt = WritingPrompt(
            text=p["text"],
            category=p["category"],
            difficulty=p["difficulty"],
            active=True,
        )
        db.add(prompt)
        inserted += 1

    db.commit()
    return inserted
