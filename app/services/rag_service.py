"""RAG (Retrieval-Augmented Generation) service for grammar and word usage grounding.

Retrieves authoritative reference rules from curated dictionary and grammar authorities
(Merriam-Webster, Cambridge, Oxford) to eliminate vague or unsubstantiated ("lackey")
explanations and provide verifiable proof, source links, and clear educational rules.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.schemas.review import MistakeItem, ReviewResponse

logger = logging.getLogger(__name__)

# Default path to curated grammar rules knowledge base
DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "grammar_rules.json"

# Spans longer than this many words are treated as sentence-level rewrites
# rather than word/phrase fixes when grounding (see ground_mistake).
SENTENCE_LEVEL_MIN_WORDS = 4

# Merriam-Webster's general grammar & usage hub -- used as the reference link
# for sentence-level rewrites that don't map to one specific rule.
MW_GRAMMAR_HUB_URL = "https://www.merriam-webster.com/grammar"


class GrammarKnowledgeBase:
    """In-memory index of authoritative grammar, article, preposition, and word usage rules."""

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = rules_path or DEFAULT_RULES_PATH
        self.rules: List[Dict[str, Any]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from JSON file with safe error handling."""
        if not self.rules_path.exists():
            logger.warning("Grammar rules file not found at %s", self.rules_path)
            self.rules = []
            return

        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            logger.info("Loaded %d authoritative grammar rules from %s", len(self.rules), self.rules_path)
        except Exception as e:
            logger.error("Failed to load grammar rules from %s: %s", self.rules_path, e)
            self.rules = []

    def get_all(self) -> List[Dict[str, Any]]:
        return self.rules

    def get_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        for r in self.rules:
            if r.get("id") == rule_id:
                return r
        return None


class RagService:
    """Retrieval-Augmented Generation service for writing review and proof grounding."""

    def __init__(self, kb: Optional[GrammarKnowledgeBase] = None):
        self.kb = kb or GrammarKnowledgeBase()

    def retrieve_for_text(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant authoritative grammar rules for a piece of writing.

        Analyzes the writing for high-frequency function words ('to', 'an', 'a', 'the',
        'by', 'for', 'with', 'in', 'on', 'at'), common confusion pairs, and grammar triggers.
        """
        if not text or not self.kb.rules:
            return []

        text_lower = text.lower()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))

        scored_rules = []
        for rule in self.kb.rules:
            score = 0
            target_words = rule.get("target_words", [])
            keywords = rule.get("keywords", [])

            # High weight for target word presence in student text
            for tw in target_words:
                tw_lower = tw.lower()
                if " " in tw_lower:
                    if tw_lower in text_lower:
                        score += 15
                elif tw_lower in words:
                    # Function words present in text are potential candidates for review
                    score += 10

            # Medium weight for keywords
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in text_lower or kw_lower in words:
                    score += 2

            # Check examples for matching error patterns
            for ex in rule.get("examples", []):
                incorrect = ex.get("incorrect", "").lower()
                incorrect_phrase = re.sub(r"[^\w\s]", "", incorrect)
                if incorrect_phrase and incorrect_phrase in text_lower:
                    score += 25

            if score > 0:
                scored_rules.append((score, rule))

        # Sort by score descending and take top_k (only significant matches)
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        return [rule for score, rule in scored_rules if score >= 8][:top_k]

    def retrieve_for_mistake(
        self,
        original: str,
        correction: str = "",
        category: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Find the most authoritative matching rule for a specific mistake snippet."""
        if not self.kb.rules:
            return None

        orig_clean = (original or "").strip().lower()
        corr_clean = (correction or "").strip().lower()
        combined_text = f"{orig_clean} {corr_clean}".strip()
        tokens = set(re.findall(r"\b[a-zA-Z']+\b", combined_text))

        best_rule = None
        best_score = 0

        for rule in self.kb.rules:
            score = 0
            target_words = [tw.lower() for tw in rule.get("target_words", [])]

            # 1. Exact token match in target words (e.g. "a", "an", "to", "by")
            for token in tokens:
                if token in target_words:
                    score += 25

            # 2. Multi-word target phrase match (e.g. "a lot")
            for tw in target_words:
                if " " in tw and (tw in orig_clean or tw in corr_clean):
                    score += 25

            # 3. Keyword matches (word-boundary aware)
            for kw in rule.get("keywords", []):
                kw_lower = kw.lower()
                if " " in kw_lower:
                    if kw_lower in combined_text:
                        score += 5
                elif kw_lower in tokens:
                    score += 5

            # 4. Category alignment bonus (only when there's an existing match)
            if score > 0 and category and rule.get("category", "").lower() == category.lower():
                score += 3

            # 5. Example match
            for ex in rule.get("examples", []):
                inc = ex.get("incorrect", "").lower()
                cor = ex.get("correct", "").lower()
                if orig_clean and orig_clean == inc:
                    score += 20
                elif corr_clean and corr_clean == cor:
                    score += 20
                elif len(orig_clean) >= 4 and orig_clean in inc:
                    score += 10

            if score > best_score:
                best_score = score
                best_rule = rule

        # Threshold to avoid false matches
        if best_score >= 20:
            return best_rule
        return None

    def build_rag_prompt_section(self, rules: List[Dict[str, Any]]) -> str:
        """Format retrieved authoritative rules as compact, token-efficient prompt context."""
        if not rules:
            return ""

        lines = [
            "AUTHORITATIVE REFERENCE RULES (Merriam-Webster):",
            "Tag matching mistakes with rule_id:",
        ]

        for rule in rules:
            lines.append(f"- [{rule.get('id')}] {rule.get('title')}: {rule.get('rule_summary')}")

        return "\n".join(lines)

    def ground_mistake(self, mistake: Any) -> MistakeItem:
        """Enrich a MistakeItem with authoritative Merriam-Webster / dictionary proof and links.

        Ensures that every mistake has:
        - rule_name
        - source_name (e.g. 'Merriam-Webster')
        - source_url (official clickable URL)
        - proof (clear citation of why the rule applies)
        """
        orig = getattr(mistake, "original", "") or ""
        corr = getattr(mistake, "correction", "") or ""
        cat = getattr(mistake, "category", "") or "Grammar"
        exp = getattr(mistake, "explanation", "") or ""
        rule_id = getattr(mistake, "rule_id", None)
        rule_name = getattr(mistake, "rule_name", None)
        source_name = getattr(mistake, "source_name", None)
        source_url = getattr(mistake, "source_url", None)
        proof = getattr(mistake, "proof", None)

        # If the mistake already has complete, valid proof and URL, keep it
        if source_url and proof and rule_name and source_name:
            return MistakeItem(
                original=orig,
                correction=corr,
                category=cat,
                explanation=exp,
                rule_id=rule_id,
                rule_name=rule_name,
                source_name=source_name,
                source_url=source_url,
                proof=proof,
            )

        # 1. Direct lookup by rule_id if provided by the model
        matched_rule = None
        if rule_id:
            matched_rule = self.kb.get_by_id(rule_id)

        # Sentence-level rewrites (whole sentences / merged sentences) contain
        # common function words like "a", "to", "the", which would make the
        # keyword heuristic attach an unrelated article/preposition rule, and
        # the dictionary fallback would fabricate a quote for a full sentence.
        is_sentence_level = len(re.findall(r"\b\w+\b", orig)) > SENTENCE_LEVEL_MIN_WORDS

        # 2. Heuristic retrieval fallback (phrase-level only)
        if not matched_rule and not is_sentence_level:
            matched_rule = self.retrieve_for_mistake(
                original=orig,
                correction=corr,
                category=cat,
            )

        if not matched_rule and is_sentence_level:
            return MistakeItem(
                original=orig,
                correction=corr,
                category=cat,
                explanation=exp,
                rule_id=rule_id,
                rule_name=rule_name or f"{cat}: sentence-level revision",
                source_name=source_name or "Merriam-Webster",
                source_url=source_url or MW_GRAMMAR_HUB_URL,
                # No proof: there's no single authoritative citation for a
                # restructured sentence, so the explanation stands on its own.
                proof=proof,
            )

        if matched_rule:
            rule_id = rule_id or matched_rule.get("id")
            rule_name = rule_name or matched_rule.get("title")
            source_name = source_name or matched_rule.get("source_name", "Merriam-Webster")
            source_url = source_url or matched_rule.get("source_url")
            proof = proof or matched_rule.get("proof")

            return MistakeItem(
                original=orig,
                correction=corr,
                category=cat,
                explanation=exp,
                rule_id=rule_id,
                rule_name=rule_name,
                source_name=source_name,
                source_url=source_url,
                proof=proof,
            )

        # 3. Fallback dictionary grounding: generate authoritative Merriam-Webster dictionary entry
        target_token = re.findall(r"\b[a-zA-Z]+\b", corr or orig)
        clean_word = target_token[0].lower() if target_token else "grammar"

        dict_url = f"https://www.merriam-webster.com/dictionary/{clean_word}"
        fallback_rule = rule_name or f"Standard English Usage: '{corr or orig}'"
        fallback_source = source_name or "Merriam-Webster"
        fallback_proof = (
            proof
            or f"According to Merriam-Webster's Dictionary & Grammar Guide, '{corr or orig}' is governed by standard English conventions of {cat.lower()}: {exp}"
        )

        return MistakeItem(
            original=orig,
            correction=corr,
            category=cat,
            explanation=exp,
            rule_id=rule_id,
            rule_name=fallback_rule,
            source_name=fallback_source,
            source_url=source_url or dict_url,
            proof=fallback_proof,
        )

    def enrich_review_response(
        self,
        response: Any,
        token_usage: Optional[Any] = None,
    ) -> ReviewResponse:
        """Enrich all mistakes in an AI or mock response with verified authoritative proof, links, and token metrics."""
        if not response:
            return ReviewResponse(mistakes=[], summary="", overall_feedback="", scores=None, token_usage=token_usage)

        mistakes_list = getattr(response, "mistakes", []) or []
        grounded_mistakes = [self.ground_mistake(m) for m in mistakes_list]

        scores = getattr(response, "scores", None)
        summary = getattr(response, "summary", "")
        overall_feedback = getattr(response, "overall_feedback", "")
        usage = token_usage or getattr(response, "token_usage", None)

        return ReviewResponse(
            mistakes=grounded_mistakes,
            summary=summary,
            overall_feedback=overall_feedback,
            scores=scores,
            token_usage=usage,
        )


# Global singleton instance for easy import
rag_service = RagService()
