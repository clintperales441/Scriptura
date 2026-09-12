"""Pydantic schemas for review request/response."""

from pydantic import BaseModel


class MistakeItem(BaseModel):
    """A single identified mistake."""
    original: str
    correction: str
    category: str
    explanation: str


class ReviewRequest(BaseModel):
    """Request sent to the review service."""
    prompt: str
    writing: str
    category: str
    difficulty: str


class ReviewResponse(BaseModel):
    """Structured response from the review service."""
    mistakes: list[MistakeItem]
    summary: str
    overall_feedback: str
