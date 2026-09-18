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

    def test_submit_writing_persists_scores(self, db):
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
        assert updated.original_grammar_score == review.scores.grammar
        assert updated.original_fluency_score == review.scores.fluency
        assert updated.original_clarity_score == review.scores.clarity
        assert updated.original_engagement_score == review.scores.engagement

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

    def test_submit_rewrite_without_review_leaves_rewrite_scores_unset(self, db):
        """rewrite_review is optional -- omitting it should never crash,
        it should just leave the rewrite_* score columns as None."""
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
        assert updated.rewrite_grammar_score is None
        assert updated.rewrite_fluency_score is None

    def test_submit_rewrite_persists_rewrite_scores(self, db):
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        original_review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="Hello world.",
                category="Everyday", difficulty="Beginner",
            )
        )
        practice_service.submit_writing(db, session.id, "Hello world.", original_review)

        rewrite_review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="Hello beautiful world.",
                category="Everyday", difficulty="Beginner",
            )
        )
        updated = practice_service.submit_rewrite(
            db, session.id, "Hello beautiful world.", rewrite_review
        )
        assert updated.rewrite_grammar_score == rewrite_review.scores.grammar
        assert updated.rewrite_fluency_score == rewrite_review.scores.fluency
        assert updated.rewrite_clarity_score == rewrite_review.scores.clarity
        assert updated.rewrite_engagement_score == rewrite_review.scores.engagement

    def test_rewrite_marks_fixed_mistakes_as_corrected(self, db):
        """A mistake whose flagged text is absent from the rewrite should
        be marked corrected_by_rewrite=True."""
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        # "alot" is a MockReviewService pattern -> flags "alot" -> "a lot"
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="I like it alot.",
                category="Everyday", difficulty="Beginner",
            )
        )
        practice_service.submit_writing(db, session.id, "I like it alot.", review)

        # Rewrite fixes the flagged mistake
        updated = practice_service.submit_rewrite(db, session.id, "I like it a lot.")

        assert updated.corrected_mistake_count == len(updated.mistakes)
        assert all(m.corrected_by_rewrite for m in updated.mistakes)

    def test_rewrite_leaves_unfixed_mistakes_uncorrected(self, db):
        """A mistake whose flagged text is still present in the rewrite
        should stay corrected_by_rewrite=False, while an unrelated mistake
        that genuinely was fixed is still marked corrected."""
        session = practice_service.create_session(
            db, prompt_id=1, category="Everyday", difficulty="Beginner"
        )
        # "i like it alot." triggers two mock mistakes: capitalize "i", and
        # "alot" -> "a lot".
        review = MockReviewService().review(
            ReviewRequest(
                prompt="test", writing="i like it alot.",
                category="Everyday", difficulty="Beginner",
            )
        )
        practice_service.submit_writing(db, session.id, "i like it alot.", review)

        # Rewrite fixes capitalization but still repeats "alot" verbatim.
        updated = practice_service.submit_rewrite(db, session.id, "I like it alot, still.")

        alot_mistake = next(m for m in updated.mistakes if m.original_text == "alot")
        capitalization_mistake = next(m for m in updated.mistakes if m.original_text == "i")

        assert alot_mistake.corrected_by_rewrite is False
        assert capitalization_mistake.corrected_by_rewrite is True
        assert updated.corrected_mistake_count == 1

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


class TestScoresDict:
    """Tests for the scores_dict template helper."""

    def test_none_when_unscored(self):
        assert practice_service.scores_dict(None, None, None, None) is None

    def test_builds_dict_with_computed_overall(self):
        result = practice_service.scores_dict(80, 60, 100, 40)
        assert result == {
            "grammar": 80,
            "fluency": 60,
            "clarity": 100,
            "engagement": 40,
            "overall": 70,
        }
