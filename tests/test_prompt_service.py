"""Tests for prompt_service."""

import pytest
from app.services import prompt_service


class TestGetRandomPrompt:
    """Tests for prompt selection."""

    def test_get_prompt_exact_match(self, db):
        prompt = prompt_service.get_random_prompt(db, "Everyday", "Beginner")
        assert prompt is not None
        assert prompt.category == "Everyday"
        assert prompt.difficulty == "Beginner"

    def test_get_prompt_all_categories(self, db):
        for category in ["Everyday", "Opinion", "Experience", "Academic / Professional", "Technical"]:
            prompt = prompt_service.get_random_prompt(db, category, "Intermediate")
            assert prompt is not None
            assert prompt.category == category

    def test_get_prompt_all_difficulties(self, db):
        for difficulty in ["Beginner", "Intermediate", "Advanced"]:
            prompt = prompt_service.get_random_prompt(db, "Opinion", difficulty)
            assert prompt is not None
            assert prompt.difficulty == difficulty

    def test_fallback_to_category(self, db):
        """If no exact match, should fallback to category-only match."""
        prompt = prompt_service.get_random_prompt(db, "Everyday", "NonExistent")
        assert prompt is not None
        assert prompt.category == "Everyday"

    def test_fallback_to_any(self, db):
        """If no category match, should fallback to any prompt."""
        prompt = prompt_service.get_random_prompt(db, "NonExistent", "NonExistent")
        assert prompt is not None

    def test_prompt_has_text(self, db):
        prompt = prompt_service.get_random_prompt(db, "Technical", "Beginner")
        assert prompt.text
        assert len(prompt.text) > 10


class TestRecommendTopicCategory:
    """Tests for weak-area-based category recommendation."""

    def test_no_breakdown_returns_none(self):
        assert prompt_service.recommend_topic_category([]) is None

    def test_fully_corrected_category_returns_none(self):
        """If everything seen so far has been fixed on rewrite, there's
        no unresolved weak spot to recommend from."""
        breakdown = [{"category": "Grammar", "total": 5, "corrected": 5, "correction_rate": 100}]
        assert prompt_service.recommend_topic_category(breakdown) is None

    def test_picks_category_with_most_unresolved_mistakes(self):
        breakdown = [
            {"category": "Grammar", "total": 10, "corrected": 9, "correction_rate": 90},  # 1 unresolved
            {"category": "Spelling", "total": 4, "corrected": 0, "correction_rate": 0},   # 4 unresolved
        ]
        result = prompt_service.recommend_topic_category(breakdown)
        assert result is not None
        weak_category, topic_category = result
        assert weak_category == "Spelling"
        assert topic_category == prompt_service.WEAK_AREA_CATEGORY_BIAS["Spelling"][0]

    def test_unknown_category_returns_none(self):
        """A category with no entry in the bias map shouldn't crash."""
        breakdown = [{"category": "Totally Unknown", "total": 5, "corrected": 0, "correction_rate": 0}]
        assert prompt_service.recommend_topic_category(breakdown) is None


class TestGetPromptForWeakArea:
    """Tests for weak-area-biased prompt selection."""

    def test_no_breakdown_returns_all_none(self, db):
        prompt, topic, weak = prompt_service.get_prompt_for_weak_area(db, "Beginner", [])
        assert prompt is None
        assert topic is None
        assert weak is None

    def test_returns_matching_prompt_and_categories(self, db):
        breakdown = [{"category": "Sentence Structure", "total": 3, "corrected": 0, "correction_rate": 0}]
        prompt, topic, weak = prompt_service.get_prompt_for_weak_area(db, "Intermediate", breakdown)

        assert prompt is not None
        assert weak == "Sentence Structure"
        assert topic == prompt_service.WEAK_AREA_CATEGORY_BIAS["Sentence Structure"][0]
        assert prompt.category == topic
