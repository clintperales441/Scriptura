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
- Identify genuine mistakes in grammar, spelling, punctuation, word choice, sentence structure, and clarity.
- Every mistake's "category" field must be exactly one of: "Grammar", "Spelling", "Punctuation", "Word Choice", "Sentence Structure", "Clarity". Pick the closest match (e.g. verb tense and subject-verb agreement issues both go under "Grammar").
- For each mistake, provide a clear, concise explanation of WHY it is wrong and how to fix it.
- Keep explanations short (1-2 sentences) but educational.
- Be encouraging — the student is learning.
- Do not fabricate mistakes that don't exist.
- Focus on the most impactful issues (up to 10 maximum).
- Provide an overall summary of the writing quality.
- If the writing is very good, say so and suggest minor improvements or advanced tips.

SCORING:
In addition to individual mistakes, score the writing as a whole on four dimensions ("scores" field), each 0-100. Anchor each score to these bands so your scoring stays consistent across different pieces of writing, rather than drifting between calls:
- 0-39: Major, frequent problems that get in the way of understanding.
- 40-59: Noticeable problems throughout; understandable but needs real work.
- 60-79: Developing -- generally solid with some recurring issues.
- 80-100: Strong -- minor or no issues in this dimension.

- grammar: correctness of grammar, verb tense, subject-verb agreement, and mechanics.
- fluency: how naturally the sentences flow and read, independent of grammar correctness.
- clarity: how clear and well-organized the ideas are -- would a reader follow this easily?
- engagement: how engaging and distinctive the voice and word choice are -- does it read as flat/generic or does it have some personality?

Score based on the writing actually submitted, not on what an ideal rewrite would look like. Do not let one bad dimension drag down another unrelated one (e.g. spelling mistakes should mainly affect grammar, not engagement).

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
