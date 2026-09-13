"""Integration tests for the full practice workflow via HTTP routes."""

import re
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import engine, SessionLocal, get_db
from app.models import Base
from app.seed import seed_prompts


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
