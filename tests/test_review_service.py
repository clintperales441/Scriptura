"""Tests for review_service (MockReviewService + AI response parsing)."""

import json
import pytest

from app.services.review_service import MockReviewService
from app.schemas.review import ReviewRequest, ReviewResponse, MistakeItem


class TestMockReviewService:
    """Tests for MockReviewService pattern matching."""

    def setup_method(self):
        self.service = MockReviewService()

    def _review(self, writing: str) -> ReviewResponse:
        return self.service.review(
            ReviewRequest(
                prompt="Test prompt",
                writing=writing,
                category="Everyday",
                difficulty="Beginner",
            )
        )

    def test_detects_lowercase_i(self):
        result = self._review("Yesterday i went to the store.")
        categories = [m.category for m in result.mistakes]
        assert "Grammar" in categories

    def test_detects_alot(self):
        result = self._review("I learned alot today.")
        categories = [m.category for m in result.mistakes]
        assert "Spelling" in categories

    def test_detects_missing_apostrophe(self):
        result = self._review("I dont know what to do.")
        categories = [m.category for m in result.mistakes]
        assert "Punctuation" in categories

    def test_detects_very(self):
        result = self._review("The food was very good and I enjoyed it very much.")
        categories = [m.category for m in result.mistakes]
        assert "Word Choice" in categories

    def test_clean_writing_no_errors(self):
        result = self._review("The weather was nice today. I enjoyed my walk.")
        # Should return a response with no pattern-matched mistakes
        assert isinstance(result, ReviewResponse)
        assert result.summary  # Should always have a summary

    def test_short_writing_feedback(self):
        result = self._review("Hello.")
        assert "short" in result.summary.lower()

    def test_medium_writing_feedback(self):
        result = self._review("I went to the park today. It was really nice. " * 3)
        assert result.summary  # Should have some summary

    def test_returns_valid_response_type(self):
        result = self._review("Some text here.")
        assert isinstance(result, ReviewResponse)
        assert isinstance(result.mistakes, list)
        assert isinstance(result.summary, str)
        assert isinstance(result.overall_feedback, str)

    def test_all_mistakes_are_valid(self):
        result = self._review("i dont know alot about this very complicated topic.")
        for mistake in result.mistakes:
            assert isinstance(mistake, MistakeItem)
            assert mistake.original
            assert mistake.correction
            assert mistake.category
            assert mistake.explanation

    def test_scores_present_and_in_range(self):
        result = self._review("i dont know alot about this very complicated topic.")
        assert result.scores is not None
        for value in (result.scores.grammar, result.scores.fluency,
                      result.scores.clarity, result.scores.engagement):
            assert 0 <= value <= 100

    def test_more_mistakes_lowers_grammar_and_clarity_scores(self):
        clean = self._review("The weather was nice today and I enjoyed my walk very much.")
        messy = self._review("i dont know alot about this very complicated topic.")
        assert messy.scores.grammar <= clean.scores.grammar
        assert messy.scores.clarity <= clean.scores.clarity

    def test_short_writing_scores_lower_fluency_and_engagement(self):
        short = self._review("Hello.")
        longer = self._review(
            "I went to the park today and had a wonderful time with my friends. "
            "We played games, talked for hours, and enjoyed the sunny weather "
            "together before heading home for dinner."
        )
        assert short.scores.fluency < longer.scores.fluency
        assert short.scores.engagement < longer.scores.engagement


class TestWritingScores:
    """Tests for the WritingScores overall-average computation."""

    def test_overall_is_average_of_four_dimensions(self):
        from app.schemas.review import WritingScores
        scores = WritingScores(grammar=80, fluency=60, clarity=100, engagement=40)
        assert scores.overall_score == 70

    def test_overall_rounds_to_nearest_whole_number(self):
        from app.schemas.review import WritingScores
        scores = WritingScores(grammar=90, fluency=90, clarity=90, engagement=91)
        assert scores.overall_score == 90  # 90.25 rounds down

    def test_overall_not_in_json_schema(self):
        """overall_score is a computed property, not an AI-provided field --
        it must not appear in the schema handed to the AI, or the AI could
        try (and fail, or drift) to supply its own value for it."""
        from app.schemas.review import WritingScores
        schema = WritingScores.model_json_schema()
        assert "overall_score" not in schema.get("properties", {})


class TestReviewResponseParsing:
    """Tests for parsing AI-style JSON responses into ReviewResponse."""

    def test_valid_response(self):
        data = {
            "mistakes": [
                {
                    "original": "I have went",
                    "correction": "I went",
                    "category": "Grammar",
                    "explanation": "Use simple past tense.",
                }
            ],
            "summary": "Good effort with some grammar issues.",
            "overall_feedback": "Keep practicing!",
        }
        response = ReviewResponse.model_validate(data)
        assert len(response.mistakes) == 1
        assert response.mistakes[0].category == "Grammar"

    def test_empty_mistakes(self):
        data = {
            "mistakes": [],
            "summary": "Great writing!",
            "overall_feedback": "No issues found.",
        }
        response = ReviewResponse.model_validate(data)
        assert len(response.mistakes) == 0

    def test_multiple_mistakes(self):
        data = {
            "mistakes": [
                {"original": "a", "correction": "b", "category": "Grammar", "explanation": "x"},
                {"original": "c", "correction": "d", "category": "Spelling", "explanation": "y"},
                {"original": "e", "correction": "f", "category": "Punctuation", "explanation": "z"},
            ],
            "summary": "Several issues found.",
            "overall_feedback": "Needs improvement.",
        }
        response = ReviewResponse.model_validate(data)
        assert len(response.mistakes) == 3

    def test_invalid_response_missing_field(self):
        data = {"mistakes": [], "summary": "Good."}  # missing overall_feedback
        with pytest.raises(Exception):
            ReviewResponse.model_validate(data)

    def test_invalid_mistake_missing_field(self):
        data = {
            "mistakes": [{"original": "a", "correction": "b"}],  # missing category + explanation
            "summary": "x",
            "overall_feedback": "y",
        }
        with pytest.raises(Exception):
            ReviewResponse.model_validate(data)

    def test_json_round_trip(self):
        """Verify JSON serialization/deserialization works correctly."""
        original = ReviewResponse(
            mistakes=[
                MistakeItem(
                    original="dont", correction="don't",
                    category="Punctuation", explanation="Add apostrophe."
                )
            ],
            summary="One issue found.",
            overall_feedback="Good overall.",
        )
        json_str = original.model_dump_json()
        parsed = ReviewResponse.model_validate_json(json_str)
        assert parsed == original
