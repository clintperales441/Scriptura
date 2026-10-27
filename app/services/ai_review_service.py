"""AI-powered review service using Gemini API.

Implements the ReviewService Protocol to provide real AI writing analysis
using Google's Gemini models with structured JSON output.
"""

import json
import logging
import os
import re
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
    AiMistakeGenerationItem,
    AiReviewGenerationResponse,
    TokenUsage,
)
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

# Multi-layer writing evaluator prompt. The previous version only asked for
# snippet-level fixes, so the model patched single words and the "corrected"
# text ended up ~90% identical to the original -- grammatical, but with the
# same choppy structure, vague wording, and logical gaps. This version makes
# the model evaluate every sentence at four layers and rewrite whole
# sentences (or adjacent groups) when the problem is structural.
REVIEW_SYSTEM_PROMPT = """You are a senior English writing coach. Your job is not just to proofread but to make the student's writing genuinely better -- clearer, more natural, and better structured -- while keeping their ideas and voice.

REVIEW IN FOUR LAYERS. Read the whole text first, then check EVERY sentence against each layer:
1. Mechanics: grammar, verb tense, agreement, articles, prepositions, spelling, punctuation, capitalization.
2. Word choice: vague or generic words, repetition, unnatural or non-native collocations ("do a decision" -> "make a decision"), weak verbs propped up by adverbs, filler, register that is too informal for the task.
3. Sentence structure: run-ons, comma splices, fragments, choppy strings of short sentences, awkward word order, wordiness/redundancy, missing or misused transitions, monotonous sentence openings.
4. Clarity & logic: unclear references ("it", "this", "they"), missing words or steps that leave a gap in meaning, ideas in a confusing order, sentences with no visible connection to the previous one.

CHOOSING THE SPAN TO FIX:
- Word/phrase level: if the sentence is otherwise sound, fix only the smallest phrase (e.g. "a apple" -> "an apple").
- Sentence level: if a sentence has a structure, clarity, or flow problem -- or has two or more problems of any kind -- set "original" to the WHOLE sentence and rewrite it completely. The rewrite must also fix every mechanical error inside it; do NOT list those smaller errors separately.
- Multi-sentence: when two or three adjacent sentences should be merged, split, or reordered to read well, set "original" to that whole stretch (verbatim, including the spaces between sentences) and give the restructured version.
- Spans must NEVER overlap. Each piece of the student's text belongs to at most one item.

WHAT COUNTS AS A REAL CORRECTION:
- It must read the way a fluent, skilled writer would naturally express the same idea -- not the original with one word swapped. If an awkward sentence still sounds awkward after your change, you have not fixed it.
- Keep the student's meaning, ideas, and point of view. Do not invent facts, examples, or opinions they did not express. You may add a connecting word or make an implied idea explicit when it fills a real gap.
- Match the target level (see DIFFICULTY GUIDANCE): lift vocabulary and structure one clear step above what the student wrote, without jumping to language they cannot learn from.
- "original" must be copied verbatim from the student's text (same casing, spelling, punctuation). "correction" is a drop-in replacement for exactly that span.

FIELDS:
- "category": the main reason for the change -- one of "Grammar", "Spelling", "Punctuation", "Word Choice", "Sentence Structure", "Clarity". For sentence-level rewrites use the dominant problem (usually "Sentence Structure" or "Clarity").
- "explanation": 1-3 sentences. Name the specific problem and the principle that makes the correction better (e.g. "These two short sentences both start with 'I' and hide the cause-and-effect; joining them with 'because' shows the connection and removes the repetition."). Never write vague feedback like "improves flow" or "better wording" without saying why.
- "rule_id": if the change is covered by one of the reference rules below, use that rule's ID (e.g. "mw_articles_a_vs_an"); otherwise null.
- "summary": 2-3 sentences on the student's main recurring weaknesses across the whole text -- patterns, not a list of individual errors.
- "overall_feedback": 2-3 concrete, prioritized things to practice next, phrased as actions (e.g. "Combine related short sentences with 'because', 'although', or 'which'.").

SCORES (0-100), anchored to bands: 0-39 major problems, 40-59 noticeable issues, 60-79 developing, 80-100 strong.
- grammar: correctness & mechanics
- fluency: natural flow and sentence variety
- clarity: organization, logical connection, ease of understanding
- engagement: voice, precision & richness of vocabulary
Text that is grammatically correct but flat, choppy, or vague must NOT score high on fluency, clarity, or engagement.

DO NOT:
- Change correct, natural phrasing just to impose your own style.
- Stop after the obvious errors -- every sentence must be considered at all four layers.
- Return a "correction" identical to its "original"."""

# Per-level guidance so corrections stretch the learner by one step instead
# of either barely touching the text or rewriting it beyond their reach.
DIFFICULTY_GUIDANCE = {
    "Beginner": (
        "Corrections should use clear, simple sentences and common everyday vocabulary. "
        "Prioritize correctness and basic connectors (and, but, because, so, when). "
        "Join choppy sentences, but avoid long or heavily embedded clauses."
    ),
    "Intermediate": (
        "Corrections should vary sentence length and structure (compound and complex sentences, "
        "relative clauses) and use precise, natural vocabulary and common collocations. "
        "Cut wordiness and add transitions wherever ideas connect."
    ),
    "Advanced": (
        "Corrections should be concise, cohesive, and idiomatic, with sophisticated but natural "
        "vocabulary, varied syntax (subordination, participial phrases, parallelism), and smooth "
        "logical transitions. Flag anything that reads as wordy, generic, or non-native."
    ),
}
 
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


def _span_pattern(snippet: str) -> str:
    """Regex for a snippet, word-boundary anchored at word edges so short
    tokens like "i" don't match inside "like" (mirrors practice_service)."""
    prefix = r"\b" if re.match(r"^\w", snippet) else ""
    suffix = r"\b" if re.search(r"\w$", snippet) else ""
    return prefix + re.escape(snippet) + suffix


def _sanitize_mistakes(
    mistakes: list[AiMistakeGenerationItem],
    writing: str,
) -> list[AiMistakeGenerationItem]:
    """Make AI mistakes safe to highlight and apply in place.

    Sentence-level rewrites and word-level fixes can collide: if the model
    rewrites a whole sentence AND separately flags a word inside it, the
    highlighter and build_corrected_text() can only apply one of them. This:
      - drops items whose `original` isn't verbatim in the writing,
      - drops no-op items (correction == original),
      - resolves overlaps by keeping the LONGER span, since a sentence-level
        rewrite is instructed to already include the smaller fixes,
      - returns items in reading order so the review pager follows the text.
    """
    if not writing:
        return mistakes

    claimed: list[tuple[int, int]] = []
    kept: list[tuple[int, AiMistakeGenerationItem]] = []

    for item in sorted(mistakes, key=lambda m: len((m.original or "").strip()), reverse=True):
        orig = (item.original or "").strip()
        corr = (item.correction or "").strip()
        if not orig or orig == corr:
            continue

        span = None
        for match in re.finditer(_span_pattern(orig), writing):
            s, e = match.span()
            if not any(s < ce and e > cs for cs, ce in claimed):
                span = (s, e)
                break

        if span is None:
            logger.info("Dropping AI mistake not found verbatim or overlapping: %r", orig[:80])
            continue

        claimed.append(span)
        kept.append((span[0], item))

    kept.sort(key=lambda pair: pair[0])
    return [item for _, item in kept]


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
            return self._parse_response(
                response.text or "", token_usage=usage, writing=request.writing
            )

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
        """Build the multi-layer review prompt augmented with level guidance and relevant rules."""
        retrieved_rules = rag_service.retrieve_for_text(request.writing, top_k=3)
        rag_section = rag_service.build_rag_prompt_section(retrieved_rules)
        level_guidance = DIFFICULTY_GUIDANCE.get(
            request.difficulty, DIFFICULTY_GUIDANCE["Intermediate"]
        )

        prompt_parts = [
            REVIEW_SYSTEM_PROMPT,
            "---",
            f"WRITING PROMPT: {request.prompt}",
            f"CATEGORY: {request.category}",
            f"DIFFICULTY: {request.difficulty}",
            f"DIFFICULTY GUIDANCE: {level_guidance}",
            f"STUDENT'S WRITING:\n\n{request.writing}",
        ]

        if rag_section:
            prompt_parts.extend(["---", rag_section])

        prompt_parts.extend([
            "---",
            "Review every sentence at all four layers, then return your feedback and scores "
            "as JSON matching the schema."
        ])

        return "\n\n".join(prompt_parts)

    def _parse_response(
        self,
        response_text: str,
        token_usage: Optional[TokenUsage] = None,
        writing: Optional[str] = None,
    ) -> ReviewResponse:
        """Parse, validate, sanitize spans, and RAG-enrich the AI response."""
        clean_text = _clean_json(response_text or "")
        data = json.loads(clean_text)
        validated = AiReviewGenerationResponse.model_validate(data)
        if writing:
            validated.mistakes = _sanitize_mistakes(validated.mistakes, writing)
        # Ground mistakes with authoritative Merriam-Webster proof and attach token usage metrics
        return rag_service.enrich_review_response(validated, token_usage=token_usage)

    def _fallback_response(self, message: str) -> ReviewResponse:
        """Return a safe fallback when AI fails."""
        return ReviewResponse(
            mistakes=[],
            summary=message,
            overall_feedback="Unable to complete AI review at this time.",
        )
