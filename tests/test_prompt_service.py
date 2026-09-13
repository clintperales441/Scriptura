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
