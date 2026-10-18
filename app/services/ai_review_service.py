"""AI-powered review service using Gemini API.

Implements the ReviewService Protocol to provide real AI writing analysis
using Google's Gemini models with structured JSON output.
"""

import json
import logging
import os
import time
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from app.schemas.review import (
    ReviewRequest,
    ReviewResponse,
    MistakeItem,
    WritingScores,
    AiReviewGenerationResponse,
    TokenUsage,
)
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

# Streamlined, token-efficient system prompt for writing review AI
REVIEW_SYSTEM_PROMPT = """You are an expert English writing tutor. Analyze the student's writing and provide educational, encouraging feedback.

RULES:
1. Identify genuine mistakes in grammar, spelling, punctuation, word choice, sentence structure, or clarity. Category must be one of: "Grammar", "Spelling", "Punctuation", "Word Choice", "Sentence Structure", "Clarity".
2. For each mistake:
   - "original": exact verbatim substring from the student's text.
   - "correction": direct replacement snippet ready to substitute in place.
   - "explanation": concise 1-2 sentence educational explanation of when and how to use the word/phrase correctly. Do not provide vague or unsubstantiated ("lackey") feedback.
   - "rule_id": if the mistake matches one of the reference rules provided below, set rule_id to that rule's ID (e.g. "mw_articles_a_vs_an"); otherwise null.
3. Anchor scores (0-100) to bands: 0-39 (major problems), 40-59 (noticeable issues), 60-79 (developing), 80-100 (strong).
   - grammar: correctness & mechanics
   - fluency: natural flow
   - clarity: organization & ease of understanding
   - engagement: voice & vocabulary
4. Only flag real errors or genuinely awkward phrasing."""
 
# Lightweight system prompt for re-scoring rewrites without burning tokens on full review
SCORE_ONLY_SYSTEM_PROMPT = """You are an expert English writing evaluator. Your task is to score a student's writing across four dimensions (0-100).

SCORING CRITERIA:
Anchor each score to these bands so scoring stays consistent:
- 0-39: Major, frequent problems that get in the way of understanding.
- 40-59: Noticeable problems throughout; understandable but needs real work.
- 60-79: Developing -- generally solid with some recurring issues.
- 80-100: Strong -- minor or no issues in this dimension.

DIMENSIONS:
- grammar: correctness of grammar, verb tense, subject-verb agreement, and mechanics.
- fluency: how naturally the sentences flow and read, independent of grammar correctness.
- clarity: how clear and well-organized the ideas are -- would a reader follow this easily?
- engagement: how engaging and distinctive the voice and word choice are -- does it read as flat/generic or does it have some personality?

Score based on the writing actually submitted, not on what an ideal rewrite would look like. Do not let one bad dimension drag down another unrelated one.
Return only the four scores as JSON matching the requested schema."""


def _clean_json(text: str) -> str:
    """Strip markdown code fences and whitespace from AI response text."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class AiReviewService:
    """Writing review service powered by Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.6-flash",
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

    def _generate_with_retry(
        self,
        contents: str,
        schema: type,
        temperature: float = 0.3,
        max_retries: int = 2,
    ):
        """Call Gemini generate_content with automatic retry on transient errors (e.g. 503/429)."""
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                return self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=schema.model_json_schema(),
                        temperature=temperature,
                    ),
                )
            except Exception as e:
                last_exception = e
                err_text = str(e).lower()
                is_transient = any(
                    sig in err_text
                    for sig in ["503", "429", "unavailable", "resource_exhausted", "high demand"]
                )
                if attempt < max_retries and is_transient:
                    wait_sec = (attempt + 1) * 1.5
                    logger.warning(
                        "Transient error from Gemini (%s). Retrying in %.1fs (attempt %d/%d)...",
                        e,
                        wait_sec,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait_sec)
                else:
                    break
        raise last_exception

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Analyze writing using Gemini and return structured feedback."""
        prompt = self._build_prompt(request)

        try:
            response = self._generate_with_retry(prompt, AiReviewGenerationResponse)
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage = TokenUsage(
                    prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                    candidates_tokens=getattr(um, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(um, "total_token_count", 0) or 0,
                )
                logger.info(
                    "AI review token usage: prompt=%d, candidates=%d, total=%d",
                    usage.prompt_tokens,
                    usage.candidates_tokens,
                    usage.total_tokens,
                )
            return self._parse_response(response.text or "", token_usage=usage)

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

    def score_only(self, request: ReviewRequest) -> Optional[WritingScores]:
        """Score writing on 4 dimensions without full mistake analysis, saving tokens and latency."""
        prompt = self._build_score_prompt(request)

        try:
            response = self._generate_with_retry(prompt, WritingScores)
            clean_text = _clean_json(response.text or "")
            data = json.loads(clean_text)
            return WritingScores.model_validate(data)

        except ValidationError as e:
            logger.error("AI score-only validation failed: %s", e)
            return None
        except Exception as e:
            logger.error("AI score-only failed: %s", e)
            return None

    def _build_score_prompt(self, request: ReviewRequest) -> str:
        """Build a streamlined prompt focused solely on scoring."""
        return (
            f"{SCORE_ONLY_SYSTEM_PROMPT}\n\n"
            f"---\n\n"
            f"WRITING PROMPT: {request.prompt}\n\n"
            f"CATEGORY: {request.category}\n"
            f"DIFFICULTY LEVEL: {request.difficulty}\n\n"
            f"STUDENT'S WRITING:\n\n{request.writing}\n\n"
            f"---\n\n"
            f"Score the writing above on grammar, fluency, clarity, and engagement. Return your scores as JSON."
        )

    def _build_prompt(self, request: ReviewRequest) -> str:
        """Build a compact, token-efficient review prompt augmented with relevant rules."""
        retrieved_rules = rag_service.retrieve_for_text(request.writing, top_k=3)
        rag_section = rag_service.build_rag_prompt_section(retrieved_rules)

        prompt_parts = [
            REVIEW_SYSTEM_PROMPT,
            "---",
            f"WRITING PROMPT: {request.prompt}",
            f"CATEGORY: {request.category}",
            f"DIFFICULTY: {request.difficulty}",
            f"STUDENT'S WRITING:\n\n{request.writing}",
        ]

        if rag_section:
            prompt_parts.extend(["---", rag_section])

        prompt_parts.extend([
            "---",
            "Return your feedback and scores as JSON matching the schema."
        ])

        return "\n\n".join(prompt_parts)

    def _parse_response(
        self,
        response_text: str,
        token_usage: Optional[TokenUsage] = None,
    ) -> ReviewResponse:
        """Parse, validate, and RAG-enrich the AI response."""
        clean_text = _clean_json(response_text or "")
        data = json.loads(clean_text)
        validated = AiReviewGenerationResponse.model_validate(data)
        # Ground mistakes with authoritative Merriam-Webster proof and attach token usage metrics
        return rag_service.enrich_review_response(validated, token_usage=token_usage)

    def _fallback_response(self, message: str) -> ReviewResponse:
        """Return a safe fallback when AI fails."""
        return ReviewResponse(
            mistakes=[],
            summary=message,
            overall_feedback="Unable to complete AI review at this time.",
        )
