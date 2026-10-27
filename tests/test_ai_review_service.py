"""Tests for the multi-layer AI review prompt and response sanitization.

These run offline: no Gemini call is made. They cover prompt construction,
overlap/verbatim sanitizing of sentence-level vs word-level spans, and
grounding behaviour for sentence-level rewrites.
"""

import json

from app.schemas.review import AiMistakeGenerationItem, MistakeItem, ReviewRequest
from app.services.ai_review_service import (
    AiReviewService,
    DIFFICULTY_GUIDANCE,
    REVIEW_SYSTEM_PROMPT,
    _sanitize_mistakes,
)
from app.services.rag_service import MW_GRAMMAR_HUB_URL, RagService


def _item(original: str, correction: str, category: str = "Grammar") -> AiMistakeGenerationItem:
    return AiMistakeGenerationItem(
        original=original, correction=correction, category=category, explanation="x"
    )


class TestReviewPrompt:
    def _service(self) -> AiReviewService:
        return AiReviewService(api_key="test-key")

    def test_prompt_covers_all_four_layers_and_sentence_level_rewrites(self):
        for marker in ("Mechanics", "Word choice", "Sentence structure", "Clarity & logic"):
            assert marker in REVIEW_SYSTEM_PROMPT
        assert "WHOLE sentence" in REVIEW_SYSTEM_PROMPT
        assert "NEVER overlap" in REVIEW_SYSTEM_PROMPT

    def test_prompt_includes_difficulty_guidance(self):
        req = ReviewRequest(prompt="P", writing="I like it.", category="Everyday", difficulty="Advanced")
        prompt = self._service()._build_prompt(req)
        assert DIFFICULTY_GUIDANCE["Advanced"] in prompt

    def test_unknown_difficulty_falls_back_to_intermediate(self):
        req = ReviewRequest(prompt="P", writing="I like it.", category="Everyday", difficulty="Expert")
        prompt = self._service()._build_prompt(req)
        assert DIFFICULTY_GUIDANCE["Intermediate"] in prompt


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
        assert "alot" not in originals  # already fixed inside the sentence rewrite

    def test_drops_non_verbatim_and_noop_items(self):
        result = _sanitize_mistakes(
            [
                _item("My friends comes", "My friends come"),  # not in text
                _item("everyday", "everyday"),  # no-op
                _item("come", "comes"),
            ],
            self.WRITING,
        )
        assert [m.original for m in result] == ["come"]

    def test_short_token_does_not_match_inside_words(self):
        # The only standalone "i" left after the sentence rewrite is at the start.
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


class TestSentenceLevelGrounding:
    def test_sentence_rewrite_gets_no_false_rule_or_fake_proof(self):
        grounded = RagService().ground_mistake(
            MistakeItem(
                original="I go to the school for buy a apple every day.",
                correction="Every day, I go to school to buy an apple.",
                category="Sentence Structure",
                explanation="Restructured for clarity.",
            )
        )
        assert grounded.rule_id is None
        assert grounded.proof is None
        assert grounded.source_url == MW_GRAMMAR_HUB_URL

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
