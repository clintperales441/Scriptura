"""AI-powered review service using Gemini API.

Implements the ReviewService Protocol to provide real AI writing analysis
using Google's Gemini models with structured JSON output, sentence-by-sentence
suggestions, word/part-of-speech/punctuation breakdowns, and RAG grounding.
"""

import difflib
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
    AiBreakdownGenerationItem,
    AiMistakeGenerationItem,
    AiReviewGenerationResponse,
    GrammaticalRole,
    ReviewRequest,
    ReviewResponse,
    TokenUsage,
    WritingScores,
)
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

# Detailed sentence-and-word grammar coach prompt.
# Instead of giving a rough, high-level summary report, the model splits the
# student's writing sentence by sentence, provides a complete natural rewrite
# for each sentence that needs work, AND breaks each sentence down by word,
# part of speech, and punctuation mark with RAG rule grounding.
REVIEW_SYSTEM_PROMPT = """You are a senior English grammar and writing coach. Never give a rough, generic summary report. Teach the student clearly by splitting their writing sentence by sentence AND breaking each revised sentence down into its words, parts of speech, and punctuation marks.

STEP 1 — EVALUATE EVERY SENTENCE IN FOUR LAYERS:
Read the writing prompt and the whole submission first, then inspect EVERY sentence (including fragments and run-ons) across all four layers:
1. Mechanics & Punctuation: sentence-initial capitalization, pronoun "I", periods/question marks/exclamation points, commas, apostrophes, contractions, spelling.
2. Parts of Speech & Grammar:
   - Noun: singular/plural agreement, countable vs. uncountable nouns, gerunds (-ing verbal nouns after verbs like "like" or "enjoy").
   - Pronoun: required explicit subject pronouns ("I", "it", "they"), subject vs. object case, clear noun antecedents.
   - Verb: finite main verb presence, subject-verb agreement, tense consistency, linking verbs ("is", "seem", "feel") vs. action verbs, infinitives ("to + verb").
   - Adjective & Adverb: precise adjectives vs. vague ones, predicate adjectives (including "-ly" adjectives such as "lively" or "friendly"), removing weak intensifiers ("very", "really"), using adverbs to modify action verbs ("well", "quickly").
   - Article / Determiner: "a" vs. "an" (by vowel/consonant sound), definite "the" vs. zero article.
   - Preposition: place/time prepositions ("in", "on", "at"), purpose/agent prepositions ("to", "for", "by", "with"), idiomatic collocations.
   - Conjunction: coordinating conjunctions (for, and, nor, but, or, yet, so) and subordinating conjunctions ("because", "although", "while", "when", "if").
   - Interjection: expressive starters or spoken fillers ("oh", "wow", "well", "hey") and their comma/exclamation punctuation or formal removal.
3. Word choice & conciseness: redundant modifiers/pleonasm (e.g. "urban city" -> "city" because a city is already urban), unnatural collocations, register suited to the prompt.
4. Sentence structure & clarity: sentence fragments missing a subject or main verb, run-ons, comma splices, choppy clauses, awkward word order, missing logical links.

STEP 2 — SPLIT MULTI-SENTENCE TEXT SENTENCE BY SENTENCE (`mistakes`):
- Create one item in `mistakes` (in reading order) for EACH sentence that has any error or needs improvement.
- Set `original` to the WHOLE verbatim sentence from the student's text (or 2 adjacent sentences ONLY when fragments/choppy sentences must merge or a run-on must split; if a long, otherwise fluent sentence has only one isolated typo, you may still show the full sentence or exact phrase, but NEVER overlap spans).
- Set `correction` to the complete, natural rewritten sentence. Preserve the student's meaning and point of view. When a student writes a fragment (e.g. "urban city because it's lively" in response to a prompt about a favorite place to eat), use the prompt context to supply the implied subject and verb ("I like eating in the city because it is lively.").
- Set `explanation` to 1-2 clear sentences stating the main structural or grammatical transformation of that sentence.

STEP 3 — SPLIT EACH SENTENCE INTO WORDS, PARTS OF SPEECH & PUNCTUATION (`breakdown`):
- Inside EVERY `mistakes` item, populate `breakdown` in reading order to explain every changed, added, removed, or structurally important word and punctuation mark.
- Split multi-word additions into distinct grammatical roles rather than lumping a whole clause into one row (for example: "I" -> Pronoun, "like" -> Verb, "eating" -> Noun [gerund], "in" -> Preposition, "the" -> Article / Determiner, "urban city" -> "city" -> Adjective, "because" -> Conjunction, "it's" -> "it is" -> Pronoun / Verb, "lively" -> Adjective, "." -> Punctuation).
- For added elements, set `original_part` to "(missing)" and `suggested_part` to the added word or punctuation mark.
- For deleted redundant/incorrect elements, set `original_part` to the removed word (or phrase like "urban city") and `suggested_part` to "(removed)" (or the tightened replacement like "city").
- For short restructured sentences, also include key retained connectors or predicates (such as subordinating conjunction "because" or predicate adjective "lively") when explaining their part of speech clarifies how the sentence works.
- In each `breakdown` item's `explanation`, explicitly name the part of speech / mechanical role and explain WHY that exact word, form, or punctuation mark is required or preferred. Never write vague phrases like "sounds better" or "improves flow" without stating the grammatical reason.
- Set `rule_id` on both the sentence item and each `breakdown` item whenever one of the AUTHORITATIVE REFERENCE RULES applies.

SUMMARY, OVERALL FEEDBACK & SCORES:
- `summary`: 2-3 sentences diagnosing recurring grammatical, part-of-speech, and structural patterns across the submission.
- `overall_feedback`: 2-3 concrete, prioritized actions naming specific parts of speech, clause structures, or punctuation habits to apply in the rewrite.
- `scores` (0-100), anchored to bands: 0-39 major problems, 40-59 noticeable issues, 60-79 developing, 80-100 strong across `grammar`, `fluency`, `clarity`, and `engagement`.

DO NOT:
- Give only a general paragraph summary without populating `breakdown` for each revised sentence.
- Change already accurate, natural sentences just to impose personal style.
- Overlap `original` spans or return a `correction` identical to `original`."""

# Per-level guidance so corrections stretch the learner by one step instead
# of either barely touching the text or rewriting it beyond their reach.
DIFFICULTY_GUIDANCE = {
    "Beginner": (
        "Corrections should use clear, complete simple or compound sentences and everyday vocabulary. "
        "Prioritize explicit subjects and verbs, articles, prepositions, capitalization, periods, and basic "
        "conjunctions (and, but, because, so, when). Explain parts of speech in accessible, beginner-friendly terms."
    ),
    "Intermediate": (
        "Corrections should vary sentence length and structure (compound and complex sentences, "
        "relative clauses) and use precise vocabulary and natural collocations. "
        "Eliminate redundancy, weak intensifiers, and comma splices, explaining clause connectors and modifiers clearly."
    ),
    "Advanced": (
        "Corrections should be concise, cohesive, and idiomatic, with sophisticated natural "
        "vocabulary, varied syntax (subordination, participial phrases, parallelism), and crisp "
        "punctuation. Flag subtle register, collocation, and pleonasm issues with rigorous grammatical explanations."
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


_PRONOUNS = {"i", "me", "my", "mine", "you", "your", "he", "him", "his", "she", "her", "it", "its", "we", "us", "our", "they", "them", "their", "who", "whom", "this", "that", "these", "those"}
_ARTICLES = {"a", "an", "the"}
_PREPOSITIONS = {"in", "on", "at", "to", "for", "by", "with", "of", "from", "about", "into", "between", "among", "during", "through", "over", "under"}
_CONJUNCTIONS = {"and", "but", "or", "nor", "so", "yet", "because", "although", "though", "since", "while", "whereas", "if", "when", "unless"}
_INTERJECTIONS = {"oh", "wow", "well", "hey", "alas", "ouch", "hmm"}
_COMMON_VERBS = {"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must", "go", "went", "walk", "like", "love", "enjoy", "eat", "come", "comes"}


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


def _infer_role(orig_tok: str, sugg_tok: str) -> GrammaticalRole:
    """Infer a sensible GrammaticalRole for synthesized fallback breakdown steps."""
    target = (sugg_tok if sugg_tok != "(removed)" else orig_tok).strip()
    low = target.lower()
    if target in {".", ",", "!", "?", ";", ":"} or (not re.search(r"\w", target) and target):
        return "Punctuation"
    if orig_tok and sugg_tok and orig_tok.lower() == sugg_tok.lower() and orig_tok != sugg_tok:
        return "Capitalization"
    if low in _ARTICLES:
        return "Article / Determiner"
    if low in _PRONOUNS or low in {"it's", "they're"}:
        return "Pronoun"
    if low in _PREPOSITIONS:
        return "Preposition"
    if low in _CONJUNCTIONS:
        return "Conjunction"
    if low in _INTERJECTIONS:
        return "Interjection"
    if low in _COMMON_VERBS or low.endswith(("ed", "ing")):
        return "Verb"
    if low.endswith("ly") and low not in {"lively", "friendly", "lovely", "lonely"}:
        return "Adverb"
    if low in {"lively", "friendly", "lovely", "good", "nice", "bad", "urban", "far", "quite"}:
        return "Adjective"
    if " " in target:
        return "Clause / Phrase"
    return "Noun"


def _ensure_breakdown(item: AiMistakeGenerationItem) -> AiMistakeGenerationItem:
    """Guarantee every mistake item has at least one structured word/punctuation breakdown step."""
    if item.breakdown:
        return item

    orig_tokens = re.findall(r"\w+(?:'\w+)?|[.!?,;:]", item.original or "")
    corr_tokens = re.findall(r"\w+(?:'\w+)?|[.!?,;:]", item.correction or "")
    matcher = difflib.SequenceMatcher(a=orig_tokens, b=corr_tokens)

    synthesized: list[AiBreakdownGenerationItem] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        orig_seg = " ".join(orig_tokens[i1:i2]) if i1 < i2 else "(missing)"
        corr_seg = " ".join(corr_tokens[j1:j2]) if j1 < j2 else "(removed)"
        role = _infer_role(orig_seg, corr_seg)
        if orig_seg == "(missing)":
            exp = f"Added '{corr_seg}' ({role.lower()}) to complete the grammatical structure: {item.explanation}"
        elif corr_seg == "(removed)":
            exp = f"Removed '{orig_seg}' ({role.lower()}) for cleaner, non-redundant phrasing: {item.explanation}"
        else:
            exp = f"Changed '{orig_seg}' to '{corr_seg}' ({role.lower()}): {item.explanation}"
        synthesized.append(
            AiBreakdownGenerationItem(
                original_part=orig_seg,
                suggested_part=corr_seg,
                part_of_speech=role,
                explanation=exp,
                rule_id=item.rule_id,
            )
        )

    if not synthesized:
        role = _infer_role(item.original, item.correction)
        synthesized.append(
            AiBreakdownGenerationItem(
                original_part=item.original,
                suggested_part=item.correction,
                part_of_speech=role,
                explanation=item.explanation,
                rule_id=item.rule_id,
            )
        )

    item.breakdown = synthesized
    return item


def _sanitize_mistakes(
    mistakes: list[AiMistakeGenerationItem],
    writing: str,
) -> list[AiMistakeGenerationItem]:
    """Make AI mistakes safe to highlight and apply in place.

    Sentence-level rewrites and word-level fixes can collide: if the model
    rewrites a whole sentence AND separately flags a word inside it, the
    highlighter and build_corrected_text() can only apply one of them (and
    the sentence item's `breakdown` already explains the individual words).
    This:
      - drops items whose `original` isn't verbatim in the writing (with a
        whitespace-normalized fallback when line breaks differ),
      - drops no-op items (correction == original),
      - resolves overlaps by keeping the LONGER span,
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

        # Fallback if student text had newlines/multiple spaces normalized by the AI
        if span is None and " " in orig:
            flex_pat = re.sub(r"\\ ", r"\\s+", _span_pattern(orig))
            for match in re.finditer(flex_pat, writing):
                s, e = match.span()
                if not any(s < ce and e > cs for cs, ce in claimed):
                    span = (s, e)
                    item.original = writing[s:e]
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
        temperature: float = 0.25,
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
        """Analyze writing using Gemini and return structured sentence + word breakdown feedback."""
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
        """Build the structured sentence + word/POS review prompt augmented with RAG rules."""
        retrieved_rules = rag_service.retrieve_for_text(request.writing, top_k=8)
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
            "Split the writing sentence by sentence, provide a complete natural suggestion for each "
            "sentence that needs improvement, break down every word/part-of-speech/punctuation change "
            "in `breakdown` with matching `rule_id`s, and return JSON matching the schema.",
        ])

        return "\n\n".join(prompt_parts)

    def _parse_response(
        self,
        response_text: str,
        token_usage: Optional[TokenUsage] = None,
        writing: Optional[str] = None,
    ) -> ReviewResponse:
        """Parse, validate, sanitize spans, ensure word breakdowns, and RAG-enrich the AI response."""
        clean_text = _clean_json(response_text or "")
        data = json.loads(clean_text)
        validated = AiReviewGenerationResponse.model_validate(data)
        if writing:
            validated.mistakes = _sanitize_mistakes(validated.mistakes, writing)
        validated.mistakes = [_ensure_breakdown(m) for m in validated.mistakes]
        # Ground both sentence mistakes and every word/POS/punctuation breakdown item with RAG proof
        return rag_service.enrich_review_response(validated, token_usage=token_usage)

    def _fallback_response(self, message: str) -> ReviewResponse:
        """Return a safe fallback when AI fails."""
        return ReviewResponse(
            mistakes=[],
            summary=message,
            overall_feedback="Unable to complete AI review at this time.",
        )
