"""Pydantic schemas for review request/response."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Fixed set of mistake categories. Constraining this (rather than letting the
# AI free-text a category per mistake) is what makes category-based
# aggregation in progress_service.get_mistake_breakdown() meaningful --
# otherwise "Verb Tense" and "Grammar" would be counted as unrelated buckets
# even though they're the same underlying weakness.
MistakeCategory = Literal[
    "Grammar",
    "Spelling",
    "Punctuation",
    "Word Choice",
    "Sentence Structure",
    "Clarity",
]

# Explicit grammatical / mechanical roles used when breaking a sentence
# rewrite down into individual words, phrases, and punctuation marks.
GrammaticalRole = Literal[
    "Noun",
    "Pronoun",
    "Verb",
    "Adjective",
    "Adverb",
    "Preposition",
    "Conjunction",
    "Interjection",
    "Article / Determiner",
    "Punctuation",
    "Capitalization",
    "Spelling",
    "Clause / Phrase",
]


def _require_breakdown_in_schema(schema: dict) -> None:
    """Ensure `breakdown` is marked required in the JSON schema handed to Gemini,
    while allowing Python callers/tests to omit it when constructing items manually."""
    req = schema.setdefault("required", [])
    if "breakdown" not in req:
        req.append("breakdown")


class WordBreakdownItem(BaseModel):
    """A single word, part-of-speech, or punctuation breakdown step grounded with RAG evidence."""
    original_part: str
    suggested_part: str
    part_of_speech: str
    explanation: str
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    proof: Optional[str] = None


class MistakeItem(BaseModel):
    """A single identified mistake (sentence or phrase) with structured word breakdown and RAG proof."""
    original: str
    correction: str
    category: MistakeCategory
    explanation: str
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    proof: Optional[str] = None
    breakdown: list[WordBreakdownItem] = Field(default_factory=list)


class AiBreakdownGenerationItem(BaseModel):
    """Token-efficient word/POS/punctuation breakdown item for AI generation."""
    original_part: str = Field(
        description=(
            "Exact word, short phrase, or punctuation from `original` being changed or explained, "
            "or '(missing)' when a missing word or punctuation mark was added."
        )
    )
    suggested_part: str = Field(
        description=(
            "Exact replacement word, short phrase, or punctuation in `correction`, "
            "or '(removed)' when a redundant or erroneous word was deleted."
        )
    )
    part_of_speech: GrammaticalRole = Field(
        description="Specific part of speech or mechanics category of this element."
    )
    explanation: str = Field(
        description=(
            "1-2 sentences explaining WHY this specific word, part of speech, grammatical form, "
            "or punctuation mark is used, changed, or removed in this sentence."
        )
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="ID of the matching authoritative reference rule if applicable.",
    )


class AiMistakeGenerationItem(BaseModel):
    """Token-efficient schema for AI generation without verbose repetitive proof text."""
    model_config = ConfigDict(json_schema_extra=_require_breakdown_in_schema)

    original: str = Field(
        description=(
            "Exact verbatim sentence (or 2 adjacent sentences when they must merge/split, or a phrase) "
            "from the student text. Split multi-sentence writing sentence by sentence. Must never overlap."
        )
    )
    correction: str = Field(
        description=(
            "Complete, natural drop-in replacement sentence for `original` that fixes all grammatical, "
            "word-choice, structural, and punctuation issues while preserving the student's meaning."
        )
    )
    category: MistakeCategory
    explanation: str = Field(
        description=(
            "1-2 sentences summarizing the main structural or grammatical reason this sentence was revised."
        )
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="Optional ID of the primary matching authoritative rule if applicable.",
    )
    breakdown: list[AiBreakdownGenerationItem] = Field(
        default_factory=list,
        description=(
            "Step-by-step word, part-of-speech, and punctuation breakdown in reading order, "
            "explaining every added, removed, replaced, or structurally important word and punctuation mark."
        ),
    )


class TokenUsage(BaseModel):
    """Token metrics for AI call observation."""
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0


class WritingScores(BaseModel):
    """Holistic quality scores (0-100) for a piece of writing.

    Loosely modeled on the 6+1 Trait Writing framework used in writing
    education: grammar ~ Conventions, fluency ~ Sentence Fluency,
    clarity ~ Ideas/Organization, engagement ~ Voice. The AI scores each
    of these directly -- they're holistic judgments (does this read
    naturally? is the voice engaging?) that can't be reliably derived
    from a list of discrete point-fixes the way a mistake count can.

    `overall` is intentionally NOT an AI-provided field. It's always the
    average of the four dimensions (see `overall_score`), computed here
    in Python, so it can never disagree with its own components the way
    an independently-asked-for number could.
    """
    grammar: int = Field(ge=0, le=100, description="Correctness of grammar, verb tense, agreement, mechanics.")
    fluency: int = Field(ge=0, le=100, description="How naturally the sentences flow and read.")
    clarity: int = Field(ge=0, le=100, description="How clear and well-organized the ideas are.")
    engagement: int = Field(ge=0, le=100, description="How engaging and distinctive the voice/word choice is.")

    @property
    def overall_score(self) -> int:
        """Average of the four dimensions, rounded to the nearest whole
        number. A plain Python property (not a model field) so it never
        appears in the JSON schema handed to the AI -- the AI supplies
        the four components only."""
        return round((self.grammar + self.fluency + self.clarity + self.engagement) / 4)


class ReviewRequest(BaseModel):
    """Request sent to the review service."""
    prompt: str
    writing: str
    category: str
    difficulty: str


class AiReviewGenerationResponse(BaseModel):
    """Token-efficient response schema for the AI review model."""
    mistakes: list[AiMistakeGenerationItem]
    summary: str
    overall_feedback: str
    scores: Optional[WritingScores] = None


class ReviewResponse(BaseModel):
    """Structured response from the review service."""
    mistakes: list[MistakeItem]
    summary: str
    overall_feedback: str
    # Optional so a technical-failure fallback response can omit scoring
    # entirely instead of showing a misleading 0 for something that was
    # never actually assessed.
    scores: Optional[WritingScores] = None
    token_usage: Optional[TokenUsage] = None
