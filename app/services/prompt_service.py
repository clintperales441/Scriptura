"""Prompt service — query and select writing prompts."""

import random

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.writing_prompt import WritingPrompt

# Heuristic mapping from a mistake category to the topic categories most
# likely to exercise it further. This is a judgment call, not something
# derived from data -- e.g. "Sentence Structure" issues are assumed to
# show up more in analytical writing (Academic/Technical) than in short
# everyday narration. Ordered by best fit first; tune freely as you learn
# what actually helps.
WEAK_AREA_CATEGORY_BIAS: dict[str, list[str]] = {
    "Grammar": ["Opinion", "Academic / Professional", "Experience"],
    "Spelling": ["Technical", "Academic / Professional"],
    "Punctuation": ["Opinion", "Academic / Professional"],
    "Word Choice": ["Technical", "Academic / Professional"],
    "Sentence Structure": ["Academic / Professional", "Technical", "Opinion"],
    "Clarity": ["Technical", "Academic / Professional"],
}


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


def recommend_topic_category(mistake_breakdown: list[dict]) -> tuple[str, str] | None:
    """Pick a topic category to serve next, biased toward whichever
    mistake category has the most *unresolved* mistakes (seen but not
    yet fixed on rewrite) across recent sessions.

    Returns (weak_mistake_category, recommended_topic_category), or None
    if there's no mistake history yet, or nothing currently unresolved,
    to base a recommendation on.
    """
    if not mistake_breakdown:
        return None

    def unresolved(item: dict) -> int:
        return item["total"] - item["corrected"]

    weakest = max(mistake_breakdown, key=unresolved)
    if unresolved(weakest) <= 0:
        # Everything seen so far has been getting fixed on rewrite --
        # no clear weak spot to bias toward.
        return None

    topic_choices = WEAK_AREA_CATEGORY_BIAS.get(weakest["category"])
    if not topic_choices:
        return None

    return weakest["category"], topic_choices[0]


def get_prompt_for_weak_area(
    db: Session,
    difficulty: str,
    mistake_breakdown: list[dict],
) -> tuple[WritingPrompt | None, str | None, str | None]:
    """Pick a prompt biased toward the learner's current weakest mistake
    category.

    Returns (prompt, chosen_topic_category, weak_mistake_category). All
    three are None when there's no mistake history to recommend from --
    callers should fall back to the normal manual-selection flow in
    that case.
    """
    recommendation = recommend_topic_category(mistake_breakdown)
    if recommendation is None:
        return None, None, None

    weak_category, topic_category = recommendation
    prompt = get_random_prompt(db, topic_category, difficulty)
    if not prompt:
        return None, None, None

    return prompt, topic_category, weak_category
