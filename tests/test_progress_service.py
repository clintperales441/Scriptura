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


class TestGetMistakeBreakdown:
    """Tests for the recurring-mistake-category breakdown."""

    def test_empty_breakdown(self, db):
        assert progress_service.get_mistake_breakdown(db) == []

    def test_breakdown_after_session_with_mistakes(self, db):
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="i like it alot and dont regret it.",
                category="Everyday", difficulty="Beginner",
            )
        )
        practice_service.submit_writing(
            db, session.id, "i like it alot and dont regret it.", review
        )
        practice_service.submit_rewrite(db, session.id, "I like it a lot and don't regret it.")

        breakdown = progress_service.get_mistake_breakdown(db)
        assert len(breakdown) > 0
        for item in breakdown:
            assert "category" in item
            assert "total" in item
            assert "corrected" in item
            assert "correction_rate" in item
            assert item["total"] > 0
            assert 0 <= item["correction_rate"] <= 100

    def test_breakdown_reflects_correction_rate(self, db):
        """A category that's fully fixed on rewrite should show 100%."""
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="I like it alot.",
                category="Everyday", difficulty="Beginner",
            )
        )
        practice_service.submit_writing(db, session.id, "I like it alot.", review)
        practice_service.submit_rewrite(db, session.id, "I like it a lot.")

        breakdown = progress_service.get_mistake_breakdown(db)
        spelling = next(item for item in breakdown if item["category"] == "Spelling")
        assert spelling["correction_rate"] == 100

    def test_breakdown_respects_incomplete_sessions(self, db):
        """Mistakes from a session that was never completed (no rewrite
        submitted) should not appear in the breakdown."""
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="I like it alot.",
                category="Everyday", difficulty="Beginner",
            )
        )
        practice_service.submit_writing(db, session.id, "I like it alot.", review)
        # No submit_rewrite call -> session stays incomplete

        assert progress_service.get_mistake_breakdown(db) == []


def _complete_scored_session(db, writing="Hello world, this is a test."):
    """Create and fully complete one session (write + rewrite), so its
    original_* score columns get populated for score-trend tests."""
    session = practice_service.create_session(
        db, prompt_id=1, category="Everyday", difficulty="Beginner"
    )
    review = MockReviewService().review(
        ReviewRequest(prompt="test", writing=writing, category="Everyday", difficulty="Beginner")
    )
    practice_service.submit_writing(db, session.id, writing, review)
    return practice_service.submit_rewrite(db, session.id, writing)


class TestGetScoreTrend:
    """Tests for the raw score-trend data (pre-chart-geometry)."""

    def test_empty_with_no_sessions(self, db):
        assert progress_service.get_score_trend(db) == []

    def test_returns_one_point_per_scored_session(self, db):
        _complete_scored_session(db)
        _complete_scored_session(db)

        trend = progress_service.get_score_trend(db)
        assert len(trend) == 2
        for point in trend:
            assert "grammar" in point
            assert "overall" in point

    def test_oldest_first_ordering(self, db):
        """The trend should read left-to-right as earlier -> now, so it
        must be oldest-first even though the underlying query orders by
        completed_at descending to apply the window limit."""
        first = _complete_scored_session(db, "First session writing sample.")
        second = _complete_scored_session(db, "Second session writing sample.")

        trend = progress_service.get_score_trend(db)
        assert trend[0]["completed_at"] <= trend[-1]["completed_at"]

    def test_respects_session_window(self, db):
        for _ in range(5):
            _complete_scored_session(db)

        trend = progress_service.get_score_trend(db, session_window=3)
        assert len(trend) == 3


class TestGetScoreTrendChart:
    """Tests for the SVG-ready trend chart geometry."""

    def test_none_with_fewer_than_two_points(self, db):
        assert progress_service.get_score_trend_chart(db) is None
        _complete_scored_session(db)
        assert progress_service.get_score_trend_chart(db) is None

    def test_chart_with_two_or_more_points(self, db):
        _complete_scored_session(db)
        _complete_scored_session(db)

        chart = progress_service.get_score_trend_chart(db)
        assert chart is not None
        assert chart["session_count"] == 2
        assert "grammar" in chart["lines"]
        assert "overall" in chart["lines"]
        # Each line should have as many x,y pairs as sessions
        assert len(chart["lines"]["grammar"].split()) == 2

    def test_latest_matches_most_recent_session(self, db):
        _complete_scored_session(db, "First session writing sample.")
        second = _complete_scored_session(db, "Second session writing sample.")

        chart = progress_service.get_score_trend_chart(db)
        assert chart["latest"]["grammar"] == second.original_grammar_score
