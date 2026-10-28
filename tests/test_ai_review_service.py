"""Tests for structured sentence + word/POS/punctuation AI review and RAG grounding.

These run offline: no live Gemini call is made. They cover prompt construction,
JSON schema requirements for `breakdown`, span sanitization, fallback breakdown
synthesis, and purposeful RAG grounding across all parts of speech and punctuation.
"""

import json

from app.schemas.review import (
    AiBreakdownGenerationItem,
    AiMistakeGenerationItem,
    AiReviewGenerationResponse,
    MistakeItem,
    ReviewRequest,
    WordBreakdownItem,
)
from app.services.ai_review_service import (
    AiReviewService,
    DIFFICULTY_GUIDANCE,
    REVIEW_SYSTEM_PROMPT,
    _sanitize_mistakes,
)
from app.services.rag_service import RagService
from app.services.review_service import MockReviewService


def _item(original: str, correction: str, category: str = "Grammar") -> AiMistakeGenerationItem:
    return AiMistakeGenerationItem(
        original=original, correction=correction, category=category, explanation="x"
    )


class TestReviewPrompt:
    def _service(self) -> AiReviewService:
        return AiReviewService(api_key="test-key")

    def test_prompt_covers_layers_parts_of_speech_and_word_breakdown(self):
        for marker in (
            "Mechanics & Punctuation",
            "Parts of Speech & Grammar",
            "Word choice & conciseness",
            "Sentence structure & clarity",
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
            "breakdown",
        ):
            assert marker in REVIEW_SYSTEM_PROMPT
        assert "WHOLE verbatim sentence" in REVIEW_SYSTEM_PROMPT
        assert "NEVER overlap" in REVIEW_SYSTEM_PROMPT

    def test_prompt_retrieves_structural_and_redundancy_rag_rules(self):
        req = ReviewRequest(
            prompt="Describe your favorite place to eat.",
            writing="urban city because it's lively",
            category="Everyday",
            difficulty="Beginner",
        )
        prompt = self._service()._build_prompt(req)
        assert "mw_sentence_fragments" in prompt
        assert "mw_redundancy_pleonasm" in prompt
        assert "mw_punctuation_end_marks_caps" in prompt
        assert "mw_pos_conjunction_clauses" in prompt

    def test_prompt_includes_difficulty_guidance(self):
        req = ReviewRequest(prompt="P", writing="I like it.", category="Everyday", difficulty="Advanced")
        prompt = self._service()._build_prompt(req)
        assert DIFFICULTY_GUIDANCE["Advanced"] in prompt

    def test_unknown_difficulty_falls_back_to_intermediate(self):
        req = ReviewRequest(prompt="P", writing="I like it.", category="Everyday", difficulty="Expert")
        prompt = self._service()._build_prompt(req)
        assert DIFFICULTY_GUIDANCE["Intermediate"] in prompt


class TestStructuredBreakdownSchema:
    def test_gemini_json_schema_requires_breakdown_field(self):
        schema = AiReviewGenerationResponse.model_json_schema()
        defs = schema.get("$defs", {})
        mistake_schema = defs.get("AiMistakeGenerationItem", {})
        assert "breakdown" in mistake_schema.get("required", [])
        assert "AiBreakdownGenerationItem" in defs

    def test_parse_response_synthesizes_fallback_breakdown_when_omitted(self):
        service = AiReviewService(api_key="test-key")
        writing = "She bought a apple yesterday"
        payload = {
            "mistakes": [
                {
                    "original": "She bought a apple yesterday",
                    "correction": "She bought an apple yesterday.",
                    "category": "Grammar",
                    "explanation": "Use 'an' before vowel sounds and end with a period.",
                }
            ],
            "summary": "Check articles and end punctuation.",
            "overall_feedback": "Use 'an' before vowel sounds.",
        }
        resp = service._parse_response(json.dumps(payload), writing=writing)
        assert len(resp.mistakes) == 1
        mistake = resp.mistakes[0]
        assert len(mistake.breakdown) >= 2
        roles = {b.part_of_speech for b in mistake.breakdown}
        assert "Article / Determiner" in roles
        assert "Punctuation" in roles
        assert all(b.proof and b.source_url for b in mistake.breakdown)


class TestSanitizeMistakes:
    WRITING = "i go to school everyday. It is very far and i walk alot. My friend come with me."

    def test_sentence_level_span_wins_over_nested_word_fix(self):
        sentence = "It is very far and i walk alot."
        result = _sanitize_mistakes(
            [
                _item("alot", "a lot", "Spelling"),
                _item(sentence, "It is quite far, but I walk there anyway.", "Sentence Structure"),
            ],
            self.WRITING,
        )
        originals = [m.original for m in result]
        assert sentence in originals
        assert "alot" not in originals

    def test_drops_non_verbatim_and_noop_items(self):
        result = _sanitize_mistakes(
            [
                _item("My friends comes", "My friends come"),
                _item("everyday", "everyday"),
                _item("come", "comes"),
            ],
            self.WRITING,
        )
        assert [m.original for m in result] == ["come"]

    def test_recovers_span_across_newlines(self):
        multiline = "urban city\nbecause it's lively"
        result = _sanitize_mistakes(
            [_item("urban city because it's lively", "I like eating in the city because it is lively.")],
            multiline,
        )
        assert len(result) == 1
        assert result[0].original == multiline

    def test_short_token_does_not_match_inside_words(self):
        result = _sanitize_mistakes(
            [
                _item("It is very far and i walk alot.", "It is far, so I walk a lot."),
                _item("i", "I"),
            ],
            self.WRITING,
        )
        assert len(result) == 2

    def test_results_are_in_reading_order(self):
        result = _sanitize_mistakes(
            [_item("come", "comes"), _item("everyday", "every day", "Spelling")],
            self.WRITING,
        )
        assert [m.original for m in result] == ["everyday", "come"]

    def test_parse_response_applies_sanitizer(self):
        service = AiReviewService(api_key="test-key")
        payload = {
            "mistakes": [
                {"original": "alot", "correction": "a lot", "category": "Spelling", "explanation": "x"},
                {
                    "original": "It is very far and i walk alot.",
                    "correction": "It is quite far, but I walk there anyway.",
                    "category": "Sentence Structure",
                    "explanation": "x",
                },
            ],
            "summary": "s",
            "overall_feedback": "f",
        }
        resp = service._parse_response(json.dumps(payload), writing=self.WRITING)
        assert len(resp.mistakes) == 1
        assert resp.mistakes[0].category == "Sentence Structure"
        assert len(resp.mistakes[0].breakdown) >= 1


class TestPurposefulRagGrounding:
    def test_urban_city_fragment_breakdown_grounds_every_pos_and_punctuation_with_rag(self):
        result = MockReviewService().review(
            ReviewRequest(
                prompt="Describe your favorite place to eat. What do you usually order and why?",
                writing="urban city because it's lively",
                category="Everyday",
                difficulty="Beginner",
            )
        )
        assert len(result.mistakes) == 1
        m = result.mistakes[0]
        assert m.original == "urban city because it's lively"
        assert m.correction == "I like eating in the city because it is lively."
        assert m.rule_id == "mw_sentence_fragments"
        assert m.proof and "fragment" in m.proof.lower()
        assert m.source_url.startswith("https://www.merriam-webster.com/")

        # Every breakdown step has a POS role, explanation, and authoritative RAG proof
        assert len(m.breakdown) >= 8
        pos_seen = {b.part_of_speech for b in m.breakdown}
        for expected_pos in (
            "Pronoun",
            "Verb",
            "Noun",
            "Preposition",
            "Article / Determiner",
            "Adjective",
            "Conjunction",
            "Punctuation",
        ):
            assert expected_pos in pos_seen

        for b in m.breakdown:
            assert b.explanation
            assert b.source_name == "Merriam-Webster"
            assert b.source_url and b.source_url.startswith("https://www.merriam-webster.com/")
            assert b.proof

        # Verify specific RAG rule assignments on key breakdown items
        redundancy_item = next(b for b in m.breakdown if b.original_part == "urban city")
        assert redundancy_item.rule_id == "mw_redundancy_pleonasm"
        assert "pleonasm" in redundancy_item.proof.lower()

        prep_item = next(b for b in m.breakdown if b.suggested_part == "in")
        assert prep_item.rule_id == "mw_prepositions_in_on_at_place"

        punct_item = next(b for b in m.breakdown if b.part_of_speech == "Punctuation")
        assert punct_item.rule_id == "mw_punctuation_end_marks_caps"

    def test_sentence_rewrite_grounds_with_real_kb_rule_not_unrelated_article_or_fake_dict_url(self):
        rag = RagService()
        # Unchanged background 'a' and 'the' should NOT cause a false 'a vs an' match
        grounded = rag.ground_mistake(
            MistakeItem(
                original="At the cafe, a waiter smiled, we ordered tea.",
                correction="At the cafe, a waiter smiled, and we ordered tea.",
                category="Punctuation",
                explanation="Fixed comma splice by adding coordinating conjunction 'and'.",
            )
        )
        assert grounded.rule_id == "mw_run_on_comma_splice"
        assert grounded.proof is not None
        assert grounded.source_url.startswith("https://www.merriam-webster.com/")
        assert "/dictionary/at" not in grounded.source_url

    def test_sentence_rewrite_keeps_explicit_rule_id(self):
        grounded = RagService().ground_mistake(
            MistakeItem(
                original="She ate a apple at the park today.",
                correction="She ate an apple in the park today.",
                category="Grammar",
                explanation="Use 'an' before vowel sounds.",
                rule_id="mw_articles_a_vs_an",
            )
        )
        assert grounded.rule_id == "mw_articles_a_vs_an"
        assert grounded.proof
