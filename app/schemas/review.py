"""Pydantic schemas for review request/response."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Fixed set of mistake categories. Constraining this (rather than letting the
# AI free-text a category per mistake) is what makes category-based
# aggregation in progress_service.get_mistake_breakdown() meaningful --
# otherwise "Verb Tense" and "Grammar" would be counted as unrelated buckets
# even though they're the same underlying weakness.
MistakeCategory = Literal[
    "Grammar",
    "Spelling",
    "Punctuation",
    "Word Choice",
    "Sentence Structure",
    "Clarity",
]


class MistakeItem(BaseModel):
    """A single identified mistake."""
    original: str
    correction: str
    category: MistakeCategory
    explanation: str


class WritingScores(BaseModel):
    """Holistic quality scores (0-100) for a piece of writing.

    Loosely modeled on the 6+1 Trait Writing framework used in writing
    education: grammar ~ Conventions, fluency ~ Sentence Fluency,
    clarity ~ Ideas/Organization, engagement ~ Voice. The AI scores each
    of these directly -- they're holistic judgments (does this read
    naturally? is the voice engaging?) that can't be reliably derived
    from a list of discrete point-fixes the way a mistake count can.

    `overall` is intentionally NOT an AI-provided field. It's always the
    average of the four dimensions (see `overall_score`), computed here
    in Python, so it can never disagree with its own components the way
    an independently-asked-for number could.
    """
    grammar: int = Field(ge=0, le=100, description="Correctness of grammar, verb tense, agreement, mechanics.")
    fluency: int = Field(ge=0, le=100, description="How naturally the sentences flow and read.")
    clarity: int = Field(ge=0, le=100, description="How clear and well-organized the ideas are.")
    engagement: int = Field(ge=0, le=100, description="How engaging and distinctive the voice/word choice is.")

    @property
    def overall_score(self) -> int:
        """Average of the four dimensions, rounded to the nearest whole
        number. A plain Python property (not a model field) so it never
        appears in the JSON schema handed to the AI -- the AI supplies
        the four components only."""
        return round((self.grammar + self.fluency + self.clarity + self.engagement) / 4)


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
    # Optional so a technical-failure fallback response can omit scoring
    # entirely instead of showing a misleading 0 for something that was
    # never actually assessed.
    scores: Optional[WritingScores] = None
