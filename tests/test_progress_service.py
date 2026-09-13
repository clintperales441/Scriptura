"""Tests for progress_service."""

import pytest
from datetime import datetime, timezone

from app.services import progress_service, practice_service
from app.services.review_service import MockReviewService
from app.schemas.review import ReviewRequest


def _complete_session(db, category="Everyday", difficulty="Beginner"):
    """Helper: create and complete a full session."""
    session = practice_service.create_session(db, prompt_id=1, category=category, difficulty=difficulty)
    review = MockReviewService().review(
        ReviewRequest(prompt="test", writing="Some test writing here.", category=category, difficulty=difficulty)
    )
    practice_service.submit_writing(db, session.id, "Some test writing here.", review)
    practice_service.submit_rewrite(db, session.id, "Some improved test writing here.")
    return session


class TestGetStats:
    """Tests for dashboard statistics."""

    def test_empty_stats(self, db):
        stats = progress_service.get_stats(db)
        assert stats["sessions_completed"] == 0
        assert stats["total_words"] == 0
        assert stats["current_streak"] == 0
        assert stats["recent_sessions"] == []

    def test_after_one_session(self, db):
        _complete_session(db)
        stats = progress_service.get_stats(db)
        assert stats["sessions_completed"] == 1
        assert stats["total_words"] > 0
        assert len(stats["recent_sessions"]) == 1

    def test_after_multiple_sessions(self, db):
        _complete_session(db, "Everyday")
        _complete_session(db, "Opinion")
        _complete_session(db, "Technical")
        stats = progress_service.get_stats(db)
        assert stats["sessions_completed"] == 3
        assert len(stats["recent_sessions"]) == 3

    def test_recent_sessions_have_required_fields(self, db):
        _complete_session(db)
        stats = progress_service.get_stats(db)
        session = stats["recent_sessions"][0]
        assert "id" in session
        assert "category" in session
        assert "date" in session
        assert "word_count" in session
        assert "mistake_count" in session

    def test_incomplete_session_not_counted(self, db):
        """Sessions that are started but not completed should not appear in stats."""
        practice_service.create_session(db, prompt_id=1, category="Everyday", difficulty="Beginner")
        stats = progress_service.get_stats(db)
        assert stats["sessions_completed"] == 0
        assert len(stats["recent_sessions"]) == 0
