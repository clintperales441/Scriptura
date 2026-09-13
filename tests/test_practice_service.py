"""Tests for practice_service."""

import pytest
from app.services import practice_service
from app.services.practice_service import count_words
from app.services.review_service import MockReviewService
from app.schemas.review import ReviewRequest


class TestCountWords:
    """Tests for the word counting function."""

    def test_normal_text(self):
        assert count_words("Hello world") == 2

    def test_multiline(self):
        assert count_words("Hello\nworld\nfoo") == 3

    def test_extra_whitespace(self):
        assert count_words("  Hello   world  ") == 2

    def test_empty_string(self):
        assert count_words("") == 0

    def test_whitespace_only(self):
        assert count_words("   ") == 0

    def test_none(self):
        assert count_words(None) == 0

    def test_single_word(self):
        assert count_words("Hello") == 1

    def test_long_text(self):
        text = " ".join(["word"] * 500)
        assert count_words(text) == 500


class TestSessionLifecycle:
    """Tests for session create/submit/rewrite lifecycle."""

    def test_create_session(self, db):
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        assert session.id is not None
        assert session.category == "Everyday"
        assert session.difficulty == "Beginner"
        assert session.completed is False
        assert session.original_text is None

    def test_submit_writing(self, db):
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="I have went there very quickly.",
                category="Everyday", difficulty="Beginner",
            )
        )
        updated = practice_service.submit_writing(
            db, session.id, "I have went there very quickly.", review
        )
        assert updated.original_text == "I have went there very quickly."
        assert updated.original_word_count == 6
        assert updated.mistake_count > 0
        assert len(updated.mistakes) > 0

    def test_submit_rewrite(self, db):
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="Hello world.",
                category="Everyday", difficulty="Beginner",
            )
        )
        practice_service.submit_writing(db, session.id, "Hello world.", review)

        updated = practice_service.submit_rewrite(db, session.id, "Hello beautiful world.")
        assert updated.rewritten_text == "Hello beautiful world."
        assert updated.rewrite_word_count == 3
        assert updated.completed is True
        assert updated.completed_at is not None

    def test_get_nonexistent_session(self, db):
        result = practice_service.get_session(db, "nonexistent-id")
        assert result is None

    def test_submit_writing_nonexistent(self, db):
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="text",
                category="Everyday", difficulty="Beginner",
            )
        )
        result = practice_service.submit_writing(db, "nonexistent", "text", review)
        assert result is None

    def test_submit_rewrite_nonexistent(self, db):
        result = practice_service.submit_rewrite(db, "nonexistent", "text")
        assert result is None
