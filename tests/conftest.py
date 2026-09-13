"""Shared test fixtures for Scriptura tests."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.seed import seed_prompts


@pytest.fixture
def db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()

    # Seed prompts
    seed_prompts(session)

    yield session

    session.close()
    engine.dispose()
