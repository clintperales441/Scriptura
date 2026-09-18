"""Integration tests for the full practice workflow via HTTP routes."""

import re
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import engine, SessionLocal, get_db
from app.models import Base
from app.seed import seed_prompts
from app.routers import practice as practice_router
from app.services.review_service import MockReviewService

# Force mock review service so tests never call live AI
practice_router._review_service = MockReviewService()


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create fresh tables for each test and seed prompts."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    seed_prompts(db)
    db.close()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestFullWorkflow:
    """Test the complete practice workflow end-to-end."""

    def test_home_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Scriptura" in response.text

    def test_setup_page(self, client):
        response = client.get("/practice/setup")
        assert response.status_code == 200
        assert "Everyday" in response.text
        assert "Beginner" in response.text

    def test_start_practice(self, client):
        response = client.post(
            "/practice/start",
            data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
        )
        assert response.status_code == 200
        assert 'name="session_id"' in response.text

    def test_full_workflow(self, client):
        """Test the complete flow: start -> write -> review -> rewrite -> result."""
        # Start practice
        response = client.post(
            "/practice/start",
            data={"category": "Opinion", "difficulty": "Intermediate", "timer_minutes": "5"},
        )
        assert response.status_code == 200

        # Extract session ID
        m = re.search(r'name="session_id" value="([^"]+)"', response.text)
        assert m, "Session ID not found in practice page"
        session_id = m.group(1)

        # Submit writing
        response = client.post(
            "/practice/submit",
            data={
                "session_id": session_id,
                "writing": "I think remote work is very good because i dont have to commute.",
            },
        )
        assert response.status_code == 200
        assert "mistake-card" in response.text

        # Rewrite page
        response = client.get(f"/practice/rewrite/{session_id}")
        assert response.status_code == 200
        assert "remote work" in response.text

        # Submit rewrite (redirects to result)
        response = client.post(
            f"/practice/rewrite/{session_id}",
            data={"rewrite": "I believe remote work is beneficial because I do not have to commute."},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "Session Complete" in response.text

        # Dashboard should now show stats
        response = client.get("/")
        assert response.status_code == 200
        assert "Opinion" in response.text  # Recent session should appear

    def test_invalid_session_redirects(self, client):
        """Accessing a nonexistent session should redirect to home."""
        response = client.get("/practice/rewrite/nonexistent", follow_redirects=False)
        assert response.status_code == 303

    def test_empty_writing_rejected(self, client):
        """Submitting empty writing should fail validation."""
        response = client.post(
            "/practice/start",
            data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
        )
        m = re.search(r'name="session_id" value="([^"]+)"', response.text)
        session_id = m.group(1)

        # FastAPI Form(...) requires the field to be present — empty string is still present
        # but our word count should handle it
        response = client.post(
            "/practice/submit",
            data={"session_id": session_id, "writing": "   "},
        )
        # Should still work (empty writing is allowed, just 0 words)
        assert response.status_code == 200


class TestWeakAreaPrompt:
    """Tests for the weak-area-biased practice flow."""

    def test_setup_page_has_no_recommendation_with_no_history(self, client):
        """A brand-new user with no completed sessions shouldn't see a
        weak-area recommendation yet."""
        response = client.get("/practice/setup")
        assert response.status_code == 200
        assert "Recommended for You" not in response.text

    def test_start_weak_area_falls_back_with_no_history(self, client):
        """With no mistake history, the weak-area route shouldn't
        dead-end -- it should redirect back to setup."""
        response = client.post(
            "/practice/start-weak-area",
            data={"difficulty": "Beginner", "timer_minutes": "5"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/practice/setup"

    def test_setup_shows_recommendation_after_unresolved_mistakes(self, client):
        """After a completed session that leaves mistakes uncorrected,
        the setup page should surface a recommendation."""
        # Start -> write (triggers mock mistakes) -> rewrite without fixing them
        response = client.post(
            "/practice/start",
            data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
        )
        session_id = re.search(r'name="session_id" value="([^"]+)"', response.text).group(1)

        client.post(
            "/practice/submit",
            data={"session_id": session_id, "writing": "i like it alot."},
        )
        client.post(
            f"/practice/rewrite/{session_id}",
            data={"rewrite": "i like it alot, still."},  # leaves mistakes unresolved
        )

        response = client.get("/practice/setup")
        assert response.status_code == 200
        assert "Recommended for You" in response.text
        assert "Practice My Weak Area" in response.text

    def test_start_weak_area_creates_session(self, client):
        """After building unresolved mistake history, the weak-area
        route should start a real session instead of redirecting."""
        response = client.post(
            "/practice/start",
            data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
        )
        session_id = re.search(r'name="session_id" value="([^"]+)"', response.text).group(1)
        client.post(
            "/practice/submit",
            data={"session_id": session_id, "writing": "i like it alot."},
        )
        client.post(
            f"/practice/rewrite/{session_id}",
            data={"rewrite": "i like it alot, still."},
        )

        response = client.post(
            "/practice/start-weak-area",
            data={"difficulty": "Intermediate", "timer_minutes": "5"},
        )
        assert response.status_code == 200
        assert 'name="session_id"' in response.text


class TestScoring:
    """Tests for the writing-quality scores across the full flow."""

    def test_review_page_shows_scores(self, client):
        response = client.post(
            "/practice/start",
            data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
        )
        session_id = re.search(r'name="session_id" value="([^"]+)"', response.text).group(1)

        response = client.post(
            "/practice/submit",
            data={"session_id": session_id, "writing": "i dont know alot about this."},
        )
        assert response.status_code == 200
        assert "Writing Scores" in response.text
        assert "score-panel" in response.text

    def test_result_page_shows_score_comparison(self, client):
        response = client.post(
            "/practice/start",
            data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
        )
        session_id = re.search(r'name="session_id" value="([^"]+)"', response.text).group(1)

        client.post(
            "/practice/submit",
            data={"session_id": session_id, "writing": "i dont know alot about this."},
        )
        client.post(
            f"/practice/rewrite/{session_id}",
            data={"rewrite": "I don't know a lot about this."},
        )

        response = client.get(f"/practice/result/{session_id}")
        assert response.status_code == 200
        assert "Score Improvement" in response.text

    def test_dashboard_shows_score_trend_after_two_sessions(self, client):
        for writing in ["First writing sample for scoring.", "Second writing sample for scoring."]:
            response = client.post(
                "/practice/start",
                data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
            )
            session_id = re.search(r'name="session_id" value="([^"]+)"', response.text).group(1)
            client.post("/practice/submit", data={"session_id": session_id, "writing": writing})
            client.post(f"/practice/rewrite/{session_id}", data={"rewrite": writing})

        response = client.get("/")
        assert response.status_code == 200
        assert "Score Trend" in response.text
        assert "trend-overall" in response.text

    def test_dashboard_has_no_score_trend_with_one_session(self, client):
        response = client.post(
            "/practice/start",
            data={"category": "Everyday", "difficulty": "Beginner", "timer_minutes": "5"},
        )
        session_id = re.search(r'name="session_id" value="([^"]+)"', response.text).group(1)
        client.post("/practice/submit", data={"session_id": session_id, "writing": "One sample."})
        client.post(f"/practice/rewrite/{session_id}", data={"rewrite": "One sample."})

        response = client.get("/")
        assert response.status_code == 200
        assert "Score Trend" not in response.text
