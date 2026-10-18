"""Review service — Protocol and MockReviewService.

Defines the ReviewService Protocol so the application depends on an
abstraction rather than a specific AI provider. MockReviewService returns
realistic fake mistakes for development and testing.
"""

import re
from typing import Protocol, runtime_checkable

from app.schemas.review import ReviewRequest, ReviewResponse, MistakeItem, WritingScores
from app.services.rag_service import rag_service


@runtime_checkable
class ReviewService(Protocol):
    """Protocol for writing review services."""

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Analyze writing and return structured feedback."""
        ...

    def score_only(self, request: ReviewRequest) -> WritingScores | None:
        """Score writing on 4 dimensions without full mistake analysis."""
        ...


def _mock_scores(mistake_count: int, word_count: int) -> WritingScores:
    """Deterministic mock scores so dev/test runs are reproducible.

    Not meant to be a realistic quality signal -- just enough variation
    (based on how many mock mistakes fired and how much was written) to
    exercise the scoring UI without calling a real AI.
    """
    grammar = max(40, 95 - mistake_count * 12)
    clarity = max(45, 90 - mistake_count * 8)
    fluency = 80 if word_count >= 20 else 60
    engagement = 70 if word_count >= 30 else 55
    return WritingScores(grammar=grammar, fluency=fluency, clarity=clarity, engagement=engagement)


class MockReviewService:
    """Returns realistic mock review results for development/testing."""

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Return mock mistakes based on simple pattern matching."""
        writing = request.writing
        mistakes: list[MistakeItem] = []

        # Simple pattern-based mock mistakes for development
        patterns = [
            {
                "search": "i",
                "check": lambda w: bool(re.search(r"\bi\b", w)),
                "original": "i",
                "correction": "I",
                "category": "Grammar",
                "explanation": 'The pronoun "I" should always be capitalized in English.',
            },
            {
                "search": "alot",
                "check": lambda w: bool(re.search(r"\balot\b", w, re.IGNORECASE)),
                "original": "alot",
                "correction": "a lot",
                "category": "Spelling",
                "explanation": '"A lot" is always written as two words, not one.',
            },
            {
                "search": "dont",
                "check": lambda w: bool(re.search(r"\bdont\b", w, re.IGNORECASE)),
                "original": "dont",
                "correction": "don't",
                "category": "Punctuation",
                "explanation": "Contractions need an apostrophe: don't, can't, won't.",
            },
            {
                "search": "very",
                "check": lambda w: bool(re.search(r"\bvery\b", w, re.IGNORECASE)),
                "original": "very",
                "correction": "extremely",
                "category": "Word Choice",
                "explanation": (
                    '"Very" is often overused. Consider a more specific alternative: '
                    '"extremely," "remarkably," "particularly," or a stronger adjective.'
                ),
            },
            {
                "search": "a apple",
                "check": lambda w: bool(re.search(r"\ba\s+apple\b", w, re.IGNORECASE)),
                "original": "a apple",
                "correction": "an apple",
                "category": "Grammar",
                "explanation": "Use the indefinite article 'an' before words beginning with a vowel sound.",
            },
            {
                "search": "a hour",
                "check": lambda w: bool(re.search(r"\ba\s+hour\b", w, re.IGNORECASE)),
                "original": "a hour",
                "correction": "an hour",
                "category": "Grammar",
                "explanation": "Use 'an' before 'hour' because the initial 'h' is silent, producing a vowel sound.",
            },
            {
                "search": "an university",
                "check": lambda w: bool(re.search(r"\ban\s+university\b", w, re.IGNORECASE)),
                "original": "an university",
                "correction": "a university",
                "category": "Grammar",
                "explanation": "Use 'a' before 'university' because it begins with a consonant sound (/j/), not a vowel sound.",
            },
            {
                "search": "for buy",
                "check": lambda w: bool(re.search(r"\bfor\s+buy\b", w, re.IGNORECASE)),
                "original": "for buy",
                "correction": "to buy",
                "category": "Word Choice",
                "explanation": "Use 'to + verb' (infinitive) to express personal purpose or intent instead of 'for + base verb'.",
            },
            {
                "search": "cut by scissors",
                "check": lambda w: bool(re.search(r"\bby\s+scissors\b", w, re.IGNORECASE)),
                "original": "by scissors",
                "correction": "with scissors",
                "category": "Grammar",
                "explanation": "Use 'with' for instruments or tools, and reserve 'by' for the agent/doer in passive constructions.",
            },
            {
                "search": "love the nature",
                "check": lambda w: bool(re.search(r"\bthe\s+nature\b", w, re.IGNORECASE)),
                "original": "the nature",
                "correction": "nature",
                "category": "Grammar",
                "explanation": "Omit the definite article 'the' when speaking about general abstract concepts like 'nature'.",
            },
        ]

        for pattern in patterns:
            if pattern["check"](writing):
                mistakes.append(
                    MistakeItem(
                        original=pattern["original"],
                        correction=pattern["correction"],
                        category=pattern["category"],
                        explanation=pattern["explanation"],
                    )
                )

        # Always provide general feedback
        word_count = len(writing.split()) if writing.strip() else 0
        if word_count < 20:
            summary = "Your response is quite short. Try to develop your ideas more fully."
        elif word_count < 50:
            summary = "Good start! Try to add more detail and examples to strengthen your writing."
        else:
            summary = "Good effort! You wrote a substantial response. Review the feedback below."

        if not mistakes:
            summary += " No obvious issues detected by the mock reviewer — AI review will provide deeper analysis."

        response = ReviewResponse(
            mistakes=mistakes,
            summary=summary,
            overall_feedback=(
                "This feedback was generated by the review service. "
                "All corrections are grounded in authoritative Merriam-Webster and language guide proof."
            ),
            scores=_mock_scores(len(mistakes), word_count),
        )
        return rag_service.enrich_review_response(response)

    def score_only(self, request: ReviewRequest) -> WritingScores:
        """Return mock scores without full mistake analysis."""
        full_review = self.review(request)
        return full_review.scores or WritingScores(grammar=75, fluency=75, clarity=75, engagement=75)
