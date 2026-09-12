# Scriptura — English Writing Practice

A localhost English-writing practice app that helps learners improve through repeated short writing sessions.

**Philosophy:** WRITE → REVIEW → UNDERSTAND → REWRITE → TRACK

## Quick Start

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload
```

Open http://localhost:8000 in your browser.

## Tech Stack

- **Python 3.14** + **FastAPI**
- **Jinja2** templates + vanilla HTML/CSS/JS
- **SQLAlchemy** + **SQLite** (Phase 2)
- **External AI API** for writing review (Phase 3)

## Project Structure

```
app/
├── main.py          # FastAPI application
├── config.py        # Settings
├── routers/         # Route handlers
├── templates/       # Jinja2 HTML templates
├── static/          # CSS, JS assets
└── data/            # Seed data
```
