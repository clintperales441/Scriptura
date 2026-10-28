"""Tests for RAG (Retrieval-Augmented Generation) grammar grounding service."""

import pytest
from markupsafe import Markup

from app.schemas.review import MistakeItem, ReviewRequest, ReviewResponse
from app.services.rag_service import RagService, GrammarKnowledgeBase, rag_service
from app.services import practice_service


class TestGrammarKnowledgeBase:
    """Test loading and querying curated Merriam-Webster grammar rules."""

    def test_loads_authoritative_rules(self):
        kb = GrammarKnowledgeBase()
        rules = kb.get_all()
        assert len(rules) >= 10
        sources = {r.get("source_name") for r in rules}
        assert "Merriam-Webster" in sources

    def test_rules_have_proof_and_url(self):
        kb = GrammarKnowledgeBase()
        for rule in kb.get_all():
            assert rule.get("id")
            assert rule.get("title")
            assert rule.get("source_name") == "Merriam-Webster"
            assert rule.get("source_url", "").startswith("https://www.merriam-webster.com/")
            assert rule.get("proof")
            assert rule.get("rule_summary")


class TestRagServiceRetrieval:
    """Test retrieval for user text and specific mistake pairs."""

    def setup_method(self):
        self.service = RagService()

    def test_retrieve_for_text_with_articles(self):
        text = "I saw a apple and an university yesterday."
        results = self.service.retrieve_for_text(text, top_k=3)
        rule_ids = [r["id"] for r in results]
        assert "mw_articles_a_vs_an" in rule_ids

    def test_retrieve_for_text_with_prepositions(self):
        text = "I went to the store for buy food and traveled by car."
        results = self.service.retrieve_for_text(text, top_k=4)
        rule_ids = [r["id"] for r in results]
        assert "mw_preposition_to_vs_for" in rule_ids
        assert "mw_preposition_by_vs_with" in rule_ids

    def test_retrieve_for_mistake_a_vs_an(self):
        rule = self.service.retrieve_for_mistake(original="a apple", correction="an apple", category="Grammar")
        assert rule is not None
        assert rule["id"] == "mw_articles_a_vs_an"
        assert rule["source_name"] == "Merriam-Webster"
        assert "https://www.merriam-webster.com/grammar/is-it-a-or-an" in rule["source_url"]
        assert "vowel sound" in rule["proof"].lower()

    def test_retrieve_for_mistake_to_vs_for(self):
        rule = self.service.retrieve_for_mistake(original="for buy", correction="to buy", category="Word Choice")
        assert rule is not None
        assert rule["id"] == "mw_preposition_to_vs_for"
        assert rule["source_name"] == "Merriam-Webster"
        assert "infinitive" in rule["proof"].lower()

    def test_retrieve_for_mistake_by_vs_with(self):
        rule = self.service.retrieve_for_mistake(original="by scissors", correction="with scissors", category="Grammar")
        assert rule is not None
        assert rule["id"] == "mw_preposition_by_vs_with"
        assert rule["source_name"] == "Merriam-Webster"
        assert "instrument" in rule["proof"].lower() or "with" in rule["proof"].lower()


class TestRagGrounding:
    """Test enriching mistakes with authoritative Merriam-Webster proof and links."""

    def setup_method(self):
        self.service = RagService()

    def test_ground_mistake_enriches_missing_fields(self):
        raw_mistake = MistakeItem(
            original="a apple",
            correction="an apple",
            category="Grammar",
            explanation="Use an before vowel sounds.",
        )
        grounded = self.service.ground_mistake(raw_mistake)
        assert grounded.source_name == "Merriam-Webster"
        assert grounded.source_url == "https://www.merriam-webster.com/grammar/is-it-a-or-an"
        assert "vowel sound" in grounded.proof.lower()
        assert grounded.rule_name is not None

    def test_ground_mistake_dictionary_fallback_for_general_word(self):
        raw_mistake = MistakeItem(
            original="exampel",
            correction="example",
            category="Spelling",
            explanation="Check spelling of example.",
        )
        grounded = self.service.ground_mistake(raw_mistake)
        assert grounded.source_name == "Merriam-Webster"
        assert "merriam-webster.com/dictionary/example" in grounded.source_url
        assert grounded.proof is not None

    def test_enrich_review_response(self):
        resp = ReviewResponse(
            mistakes=[
                MistakeItem(original="a apple", correction="an apple", category="Grammar", explanation="Use an."),
                MistakeItem(original="for go", correction="to go", category="Word Choice", explanation="Use to."),
            ],
            summary="Review summary.",
            overall_feedback="Feedback.",
        )
        enriched = self.service.enrich_review_response(resp)
        assert len(enriched.mistakes) == 2
        for m in enriched.mistakes:
            assert m.source_name == "Merriam-Webster"
            assert m.source_url.startswith("https://www.merriam-webster.com/")
            assert m.proof


class TestRenderHighlightedText:
    """Test safe in-text word highlighting with interactive RAG data attributes."""

    def test_render_highlighted_text_wraps_mistake_words(self):
        text = "She bought a apple yesterday."
        mistakes = [
            {
                "original": "a apple",
                "correction": "an apple",
                "category": "Grammar",
                "explanation": "Use an before vowel sounds.",
                "rule_name": "When to Use 'A' vs. 'An'",
                "source_name": "Merriam-Webster",
                "source_url": "https://www.merriam-webster.com/grammar/is-it-a-or-an",
                "proof": "Use an before vowel sounds.",
            }
        ]
        result = practice_service.render_highlighted_text(text, mistakes)
        assert isinstance(result, Markup)
        html_str = str(result)
        assert 'class="ai-highlight-word"' in html_str
        assert 'data-original="a apple"' in html_str
        assert 'data-correction="an apple"' in html_str
        assert 'data-source-name="Merriam-Webster"' in html_str
        assert 'data-source-url="https://www.merriam-webster.com/grammar/is-it-a-or-an"' in html_str
        assert 'data-proof="Use an before vowel sounds."' in html_str
        assert "She bought " in html_str
        assert " yesterday." in html_str

    def test_render_highlighted_text_escapes_xss_safely(self):
        text = "Bad <script>alert(1)</script> here."
        mistakes = [
            {
                "original": "<script>",
                "correction": "code",
                "category": "Grammar",
                "explanation": "No scripts.",
            }
        ]
        result = practice_service.render_highlighted_text(text, mistakes)
        html_str = str(result)
        assert "<script>" not in html_str
        assert "&lt;script&gt;" in html_str

    def test_render_empty_or_no_mistakes(self):
        assert str(practice_service.render_highlighted_text("", [])) == ""
        clean_text = "Everything was completely perfect."
        assert str(practice_service.render_highlighted_text(clean_text, [])) == clean_text


class TestPartOfSpeechBreakdownGrounding:
    """Test purposeful RAG grounding across all parts of speech and punctuation."""

    def setup_method(self):
        self.service = RagService()

    @pytest.mark.parametrize(
        "orig,sugg,pos,exp,expected_rule_id",
        [
            ("advices", "advice", "Noun", "Uncountable noun does not take plural -s.", "mw_pos_noun_usage"),
            ("(missing)", "I", "Pronoun", "Supply explicit subject pronoun.", "mw_pos_pronoun_clarity"),
            ("taste", "tastes", "Verb", "Singular subject-verb agreement.", "mw_pos_verb_agreement_tense"),
            ("lively", "lively", "Adjective", "Predicate adjective modifying city.", "mw_pos_adjective_precision"),
            ("very", "extremely", "Adverb", "Replace weak intensifier adverb.", "mw_pos_adverb_usage"),
            ("(missing)", "in", "Preposition", "Preposition of place for a city.", "mw_prepositions_in_on_at_place"),
            ("because", "because", "Conjunction", "Subordinating conjunction introducing cause.", "mw_pos_conjunction_clauses"),
            ("Oh", "Oh,", "Interjection", "Set off mild introductory interjection with a comma.", "mw_pos_interjection_register"),
            ("(missing)", ".", "Punctuation", "End declarative sentence with a period.", "mw_punctuation_end_marks_caps"),
            ("urban city", "city", "Adjective", "Remove redundant modifier (pleonasm).", "mw_redundancy_pleonasm"),
        ],
    )
    def test_grounds_every_pos_and_mechanics_category(self, orig, sugg, pos, exp, expected_rule_id):
        grounded = self.service.ground_breakdown_item(
            {
                "original_part": orig,
                "suggested_part": sugg,
                "part_of_speech": pos,
                "explanation": exp,
            }
        )
        assert grounded.rule_id == expected_rule_id
        assert grounded.source_name == "Merriam-Webster"
        assert grounded.source_url.startswith("https://www.merriam-webster.com/")
        assert grounded.proof

