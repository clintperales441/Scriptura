"""AI-powered review service using Gemini API.

Implements the ReviewService Protocol to provide real AI writing analysis
using Google's Gemini models with structured JSON output.
"""

import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from app.schemas.review import ReviewRequest, ReviewResponse, MistakeItem

logger = logging.getLogger(__name__)

# System prompt for the writing review AI
REVIEW_SYSTEM_PROMPT = """You are an expert English writing tutor. Your task is to analyze a student's writing and provide helpful, educational feedback.

CONTEXT:
- The student is practicing English writing skills.
- They responded to a writing prompt.
- Your goal is to help them LEARN, not just correct their writing.

GUIDELINES:
- Identify genuine mistakes in grammar, spelling, punctuation, word choice, sentence structure, clarity, formality, and vocabulary.
- For each mistake, provide a clear, concise explanation of WHY it is wrong and how to fix it.
- Keep explanations short (1-2 sentences) but educational.
- Be encouraging — the student is learning.
- Do not fabricate mistakes that don't exist.
- Focus on the most impactful issues (up to 10 maximum).
- Provide an overall summary of the writing quality.
- If the writing is very good, say so and suggest minor improvements or advanced tips.

IMPORTANT: Only identify real errors or genuinely awkward phrasing. Do not nitpick style preferences unless they affect clarity."""


class AiReviewService:
    """Writing review service powered by Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.7-flash",
        timeout: int = 30,
    ):
        """Initialize the Gemini client.

        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model: Model name to use.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model = model
        self._timeout = timeout

        if not self._api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = genai.Client(api_key=self._api_key)

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Analyze writing using Gemini and return structured feedback."""
        prompt = self._build_prompt(request)

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=ReviewResponse.model_json_schema(),
                    temperature=0.3,
                ),
            )

            return self._parse_response(response.text)

        except ValidationError as e:
            logger.error("AI response validation failed: %s", e)
            return self._fallback_response(
                "The AI response could not be parsed correctly. "
                "Please try again or review your writing manually."
            )
        except Exception as e:
            logger.error("AI review failed: %s", e)
            return self._fallback_response(
                f"AI review encountered an error: {type(e).__name__}. "
                "Please try again later."
            )

    def _build_prompt(self, request: ReviewRequest) -> str:
        """Build the review prompt for the AI."""
        return (
            f"{REVIEW_SYSTEM_PROMPT}\n\n"
            f"---\n\n"
            f"WRITING PROMPT: {request.prompt}\n\n"
            f"CATEGORY: {request.category}\n"
            f"DIFFICULTY LEVEL: {request.difficulty}\n\n"
            f"STUDENT'S WRITING:\n\n{request.writing}\n\n"
            f"---\n\n"
            f"Analyze the writing above. Return your analysis as JSON."
        )

    def _parse_response(self, response_text: str) -> ReviewResponse:
        """Parse and validate the AI response."""
        data = json.loads(response_text)
        return ReviewResponse.model_validate(data)

    def _fallback_response(self, message: str) -> ReviewResponse:
        """Return a safe fallback when AI fails."""
        return ReviewResponse(
            mistakes=[],
            summary=message,
            overall_feedback="Unable to complete AI review at this time.",
        )
