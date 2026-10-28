"""Review service — Protocol and MockReviewService.

Defines the ReviewService Protocol so the application depends on an
abstraction rather than a specific AI provider. MockReviewService returns
realistic structured mistakes (including word, part-of-speech, and
punctuation breakdowns grounded with RAG proof) for development and testing.
"""

import re
from typing import Protocol, runtime_checkable

from app.schemas.review import (
    MistakeItem,
    ReviewRequest,
    ReviewResponse,
    WordBreakdownItem,
    WritingScores,
)
from app.services.rag_service import rag_service


@runtime_checkable
class ReviewService(Protocol):
    """Protocol for writing review services."""

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Analyze writing and return structured feedback."""
        ...

    def score_only(self, request: ReviewRequest) -> WritingScores | None:
        """Score writing on 4 dimensions without full mistake analysis."""
        ...


def _mock_scores(mistake_count: int, word_count: int) -> WritingScores:
    """Deterministic mock scores so dev/test runs are reproducible.

    Not meant to be a realistic quality signal -- just enough variation
    (based on how many mock mistakes fired and how much was written) to
    exercise the scoring UI without calling a real AI.
    """
    grammar = max(40, 95 - mistake_count * 12)
    clarity = max(45, 90 - mistake_count * 8)
    fluency = 80 if word_count >= 20 else 60
    engagement = 70 if word_count >= 30 else 55
    return WritingScores(grammar=grammar, fluency=fluency, clarity=clarity, engagement=engagement)


class MockReviewService:
    """Returns realistic mock review results with structured word/POS breakdowns."""

    def review(self, request: ReviewRequest) -> ReviewResponse:
        """Return mock mistakes based on pattern matching with POS/word breakdowns."""
        writing = request.writing
        mistakes: list[MistakeItem] = []

        # Full sentence-fragment example demonstrating sentence + word/POS/punctuation breakdown
        frag_match = re.search(r"urban city because it's lively\.?", writing, re.IGNORECASE)
        if frag_match:
            span_text = frag_match.group(0)
            mistakes.append(
                MistakeItem(
                    original=span_text,
                    correction="I like eating in the city because it is lively.",
                    category="Sentence Structure",
                    explanation=(
                        "This phrase is a sentence fragment lacking an independent subject and main verb, "
                        "uses the redundant modifier 'urban' before 'city', and omits terminal punctuation."
                    ),
                    rule_id="mw_sentence_fragments",
                    breakdown=[
                        WordBreakdownItem(
                            original_part="(missing)",
                            suggested_part="I",
                            part_of_speech="Pronoun",
                            explanation=(
                                "Supplies the required first-person singular subject pronoun 'I', capitalized "
                                "both as the pronoun 'I' and as the opening word of the sentence."
                            ),
                            rule_id="mw_capitalization_pronoun_i",
                        ),
                        WordBreakdownItem(
                            original_part="(missing)",
                            suggested_part="like",
                            part_of_speech="Verb",
                            explanation=(
                                "Adds the finite transitive main verb 'like' so the independent clause "
                                "expresses a complete action."
                            ),
                            rule_id="mw_pos_verb_agreement_tense",
                        ),
                        WordBreakdownItem(
                            original_part="(missing)",
                            suggested_part="eating",
                            part_of_speech="Noun",
                            explanation=(
                                "Uses the gerund ('-ing' verbal noun) 'eating' as the direct object of 'like' "
                                "to name the activity."
                            ),
                            rule_id="mw_pos_noun_usage",
                        ),
                        WordBreakdownItem(
                            original_part="(missing)",
                            suggested_part="in",
                            part_of_speech="Preposition",
                            explanation=(
                                "Adds the preposition of place 'in', which is used for enclosed geographic "
                                "areas and territories such as cities."
                            ),
                            rule_id="mw_prepositions_in_on_at_place",
                        ),
                        WordBreakdownItem(
                            original_part="(missing)",
                            suggested_part="the",
                            part_of_speech="Article / Determiner",
                            explanation=(
                                "Adds the definite article 'the' before the singular countable noun 'city' "
                                "to specify the urban setting idiomatically ('in the city')."
                            ),
                            rule_id="mw_definite_article_the",
                        ),
                        WordBreakdownItem(
                            original_part="urban city",
                            suggested_part="city",
                            part_of_speech="Adjective",
                            explanation=(
                                "Removes the adjective 'urban' because the noun 'city' already means an "
                                "urban settlement, eliminating redundancy (pleonasm)."
                            ),
                            rule_id="mw_redundancy_pleonasm",
                        ),
                        WordBreakdownItem(
                            original_part="because",
                            suggested_part="because",
                            part_of_speech="Conjunction",
                            explanation=(
                                "Retains the subordinating conjunction 'because' to link the main clause "
                                "directly to the causal reason clause without a preceding comma."
                            ),
                            rule_id="mw_pos_conjunction_clauses",
                        ),
                        WordBreakdownItem(
                            original_part="it's",
                            suggested_part="it is",
                            part_of_speech="Verb",
                            explanation=(
                                "Expands the contraction 'it's' into the pronoun 'it' (referring to 'city') "
                                "plus the singular linking verb 'is' for a clearer written tone."
                            ),
                            rule_id="mw_its_vs_its",
                        ),
                        WordBreakdownItem(
                            original_part="lively",
                            suggested_part="lively",
                            part_of_speech="Adjective",
                            explanation=(
                                "Functions as a predicate adjective after the linking verb 'is' to describe "
                                "the noun 'city' (despite ending in '-ly', 'lively' is an adjective, not an adverb)."
                            ),
                            rule_id="mw_pos_adjective_precision",
                        ),
                        WordBreakdownItem(
                            original_part="(missing)" if not span_text.endswith(".") else ".",
                            suggested_part=".",
                            part_of_speech="Punctuation",
                            explanation=(
                                "Closes the complete declarative sentence with a terminal period (full stop)."
                            ),
                            rule_id="mw_punctuation_end_marks_caps",
                        ),
                    ],
                )
            )

        # Standard pattern-based mock mistakes for development & tests
        patterns = [
            {
                "check": lambda w: bool(re.search(r"\bi\b", w)),
                "original": "i",
                "correction": "I",
                "category": "Grammar",
                "explanation": 'The pronoun "I" should always be capitalized in English.',
                "rule_id": "mw_capitalization_pronoun_i",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="i",
                        suggested_part="I",
                        part_of_speech="Pronoun",
                        explanation="The first-person singular subject pronoun 'I' is capitalized in every position in a sentence.",
                        rule_id="mw_capitalization_pronoun_i",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\balot\b", w, re.IGNORECASE)),
                "original": "alot",
                "correction": "a lot",
                "category": "Spelling",
                "explanation": '"A lot" is always written as two words, not one.',
                "rule_id": "mw_spelling_a_lot",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="alot",
                        suggested_part="a lot",
                        part_of_speech="Spelling",
                        explanation="Separate the indefinite article 'a' and the noun 'lot' into two words ('a lot') when expressing a large quantity.",
                        rule_id="mw_spelling_a_lot",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\bdont\b", w, re.IGNORECASE)),
                "original": "dont",
                "correction": "don't",
                "category": "Punctuation",
                "explanation": "Contractions need an apostrophe: don't, can't, won't.",
                "rule_id": "mw_contractions_apostrophe",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="dont",
                        suggested_part="don't",
                        part_of_speech="Punctuation",
                        explanation="Insert an apostrophe between 'n' and 't' in the verb contraction 'don't' to mark the omitted 'o' in 'do not'.",
                        rule_id="mw_contractions_apostrophe",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\bvery\b", w, re.IGNORECASE)),
                "original": "very",
                "correction": "extremely",
                "category": "Word Choice",
                "explanation": (
                    '"Very" is often overused. Consider a more specific alternative: '
                    '"extremely," "remarkably," "particularly," or a stronger adjective.'
                ),
                "rule_id": "mw_pos_adverb_usage",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="very",
                        suggested_part="extremely",
                        part_of_speech="Adverb",
                        explanation="Replaces the vague intensifier adverb 'very' with a more precise degree adverb (or pair with a stronger adjective).",
                        rule_id="mw_pos_adverb_usage",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\ba\s+apple\b", w, re.IGNORECASE)),
                "original": "a apple",
                "correction": "an apple",
                "category": "Grammar",
                "explanation": "Use the indefinite article 'an' before words beginning with a vowel sound.",
                "rule_id": "mw_articles_a_vs_an",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="a",
                        suggested_part="an",
                        part_of_speech="Article / Determiner",
                        explanation="Use 'an' instead of 'a' because the following singular count noun 'apple' starts with a vowel sound.",
                        rule_id="mw_articles_a_vs_an",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\ba\s+hour\b", w, re.IGNORECASE)),
                "original": "a hour",
                "correction": "an hour",
                "category": "Grammar",
                "explanation": "Use 'an' before 'hour' because the initial 'h' is silent, producing a vowel sound.",
                "rule_id": "mw_articles_a_vs_an",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="a",
                        suggested_part="an",
                        part_of_speech="Article / Determiner",
                        explanation="Use the indefinite article 'an' because 'hour' has a silent 'h' and opens with a vowel sound.",
                        rule_id="mw_articles_a_vs_an",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\ban\s+university\b", w, re.IGNORECASE)),
                "original": "an university",
                "correction": "a university",
                "category": "Grammar",
                "explanation": "Use 'a' before 'university' because it begins with a consonant sound (/j/), not a vowel sound.",
                "rule_id": "mw_articles_a_vs_an",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="an",
                        suggested_part="a",
                        part_of_speech="Article / Determiner",
                        explanation="Use 'a' because 'university' begins with the consonant sound /j/ ('yoo'), not a vowel sound.",
                        rule_id="mw_articles_a_vs_an",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\bfor\s+buy\b", w, re.IGNORECASE)),
                "original": "for buy",
                "correction": "to buy",
                "category": "Word Choice",
                "explanation": "Use 'to + verb' (infinitive) to express personal purpose or intent instead of 'for + base verb'.",
                "rule_id": "mw_preposition_to_vs_for",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="for",
                        suggested_part="to",
                        part_of_speech="Preposition",
                        explanation="Replace the preposition 'for' with the infinitive marker 'to' before the base verb 'buy' to express purpose.",
                        rule_id="mw_preposition_to_vs_for",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\bby\s+scissors\b", w, re.IGNORECASE)),
                "original": "by scissors",
                "correction": "with scissors",
                "category": "Grammar",
                "explanation": "Use 'with' for instruments or tools, and reserve 'by' for the agent/doer in passive constructions.",
                "rule_id": "mw_preposition_by_vs_with",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="by",
                        suggested_part="with",
                        part_of_speech="Preposition",
                        explanation="Use the preposition 'with' to indicate the physical tool ('scissors') used to perform the action.",
                        rule_id="mw_preposition_by_vs_with",
                    )
                ],
            },
            {
                "check": lambda w: bool(re.search(r"\bthe\s+nature\b", w, re.IGNORECASE)),
                "original": "the nature",
                "correction": "nature",
                "category": "Grammar",
                "explanation": "Omit the definite article 'the' when speaking about general abstract concepts like 'nature'.",
                "rule_id": "mw_definite_article_the",
                "breakdown": [
                    WordBreakdownItem(
                        original_part="the",
                        suggested_part="(removed)",
                        part_of_speech="Article / Determiner",
                        explanation="Remove the definite article 'the' before the uncountable abstract noun 'nature' when referring to it in general.",
                        rule_id="mw_definite_article_the",
                    )
                ],
            },
        ]

        if not frag_match:
            for pattern in patterns:
                if pattern["check"](writing):
                    mistakes.append(
                        MistakeItem(
                            original=pattern["original"],
                            correction=pattern["correction"],
                            category=pattern["category"],
                            explanation=pattern["explanation"],
                            rule_id=pattern["rule_id"],
                            breakdown=pattern["breakdown"],
                        )
                    )

        # Always provide general feedback
        word_count = len(writing.split()) if writing.strip() else 0
        if word_count < 20:
            summary = "Your response is quite short. Try to develop your ideas more fully."
        elif word_count < 50:
            summary = "Good start! Try to add more detail and examples to strengthen your writing."
        else:
            summary = "Good effort! You wrote a substantial response. Review the feedback below."

        if not mistakes:
            summary += " No obvious issues detected by the mock reviewer — AI review will provide deeper analysis."

        response = ReviewResponse(
            mistakes=mistakes,
            summary=summary,
            overall_feedback=(
                "Review each sentence suggestion and its part-of-speech breakdown below. "
                "All corrections are grounded in authoritative Merriam-Webster grammar and usage proof."
            ),
            scores=_mock_scores(len(mistakes), word_count),
        )
        return rag_service.enrich_review_response(response)

    def score_only(self, request: ReviewRequest) -> WritingScores:
        """Return mock scores without full mistake analysis."""
        full_review = self.review(request)
        return full_review.scores or WritingScores(grammar=75, fluency=75, clarity=75, engagement=75)
