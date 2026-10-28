"""RAG (Retrieval-Augmented Generation) service for grammar and word usage grounding.

Retrieves authoritative reference rules from curated dictionary and grammar authorities
(Merriam-Webster) to eliminate vague or unsubstantiated ("lackey") explanations and
provide verifiable proof, source links, and structured part-of-speech & punctuation
evidence at both the sentence level and individual word/punctuation breakdown level.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.schemas.review import MistakeItem, ReviewResponse, WordBreakdownItem

logger = logging.getLogger(__name__)

# Default path to curated grammar rules knowledge base
DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "grammar_rules.json"

# Spans longer than this many words are treated as sentence-level rewrites
# rather than narrow word/phrase fixes when grounding (see ground_mistake).
SENTENCE_LEVEL_MIN_WORDS = 4

# Merriam-Webster's general grammar & usage hub
MW_GRAMMAR_HUB_URL = "https://www.merriam-webster.com/grammar"

# Placeholders used in breakdown items when words/punctuation are added or removed
_PLACEHOLDER_PARTS = {
    "",
    "(missing)",
    "(added)",
    "(removed)",
    "(delete)",
    "(omit)",
    "(none)",
    "none",
    "∅",
}

_PLACE_HINTS = {
    "place", "location", "city", "country", "space", "area", "room",
    "building", "surface", "street", "address", "geographic", "enclosed",
}
_TIME_HINTS = {
    "time", "clock", "hour", "day", "days", "date", "dates", "month",
    "months", "year", "years", "season", "morning", "evening", "night",
}
_SUBORDINATORS = {
    "because", "although", "though", "since", "while", "whereas", "unless", "if", "when",
}
_FINITE_VERB_HINTS = {
    "am", "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "like", "love", "enjoy", "prefer", "think",
    "believe", "go", "went", "eat", "ate", "order", "work", "live", "want",
    "need", "make", "made", "take", "took", "get", "got", "use", "feel",
}


def _strip_placeholder(part: str) -> str:
    cleaned = (part or "").strip()
    if cleaned.lower() in _PLACEHOLDER_PARTS or (
        cleaned.startswith("(") and cleaned.endswith(")")
    ):
        return ""
    return cleaned


class GrammarKnowledgeBase:
    """In-memory index of authoritative grammar, part-of-speech, punctuation, and usage rules."""

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = rules_path or DEFAULT_RULES_PATH
        self.rules: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from JSON file with safe error handling."""
        if not self.rules_path.exists():
            logger.warning("Grammar rules file not found at %s", self.rules_path)
            self.rules = []
            self._by_id = {}
            return

        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            self._by_id = {r["id"]: r for r in self.rules if r.get("id")}
            logger.info("Loaded %d authoritative grammar rules from %s", len(self.rules), self.rules_path)
        except Exception as e:
            logger.error("Failed to load grammar rules from %s: %s", self.rules_path, e)
            self.rules = []
            self._by_id = {}

    def get_all(self) -> List[Dict[str, Any]]:
        return self.rules

    def get_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        if not rule_id:
            return None
        return self._by_id.get(rule_id)


class RagService:
    """Retrieval-Augmented Generation service for structured writing review and proof grounding."""

    def __init__(self, kb: Optional[GrammarKnowledgeBase] = None):
        self.kb = kb or GrammarKnowledgeBase()

    def retrieve_for_text(self, text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Retrieve the most relevant authoritative grammar, POS, and punctuation rules for a text.

        Combines lexical triggers (articles, prepositions, confusion pairs, redundant modifiers)
        with structural detectors (fragments, missing terminal punctuation, lowercase starts,
        subordinate clauses, contractions, and comma splices).
        """
        if not text or not self.kb.rules:
            return []

        stripped = text.strip()
        text_lower = stripped.lower()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))

        # Structural signals across the student submission
        missing_end_punct = bool(stripped) and stripped[-1] not in ".!?\"'"
        lowercase_start = bool(re.match(r"^[a-z]", stripped)) or bool(
            re.search(r"[.!?]\s+[a-z]", stripped)
        )
        has_subordinator = bool(words & _SUBORDINATORS)
        has_contraction = bool(re.search(r"\b[a-z]+'[a-z]+\b", text_lower))

        # Detect likely sentence fragments (e.g., "urban city because it's lively")
        likely_fragment = False
        for chunk in re.split(r"[.!?]+", stripped):
            chunk_clean = chunk.strip()
            if not chunk_clean:
                continue
            chunk_words = re.findall(r"\b[a-zA-Z']+\b", chunk_clean.lower())
            if not chunk_words:
                continue
            # Fragment starting with a subordinator or having no finite verb before 'because'
            if chunk_words[0] in _SUBORDINATORS and "," not in chunk_clean:
                likely_fragment = True
                break
            if "because" in chunk_words:
                idx = chunk_words.index("because")
                main_part = set(chunk_words[:idx])
                if not (main_part & _FINITE_VERB_HINTS) and not any(
                    w.endswith(("ed", "es")) for w in main_part
                ):
                    likely_fragment = True
                    break
            if len(chunk_words) <= 6 and not (set(chunk_words) & _FINITE_VERB_HINTS):
                likely_fragment = True
                break

        likely_comma_splice = bool(
            re.search(r",\s+(i|it|he|she|they|we|you|this|that)\s+\w+", text_lower)
        )

        structural_boosts: Dict[str, int] = {}
        if missing_end_punct or lowercase_start:
            structural_boosts["mw_punctuation_end_marks_caps"] = 22
        if likely_fragment:
            structural_boosts["mw_sentence_fragments"] = 26
            structural_boosts["mw_pos_pronoun_clarity"] = 14
            structural_boosts["mw_pos_verb_agreement_tense"] = 14
        if has_subordinator:
            structural_boosts["mw_pos_conjunction_clauses"] = 15
        if has_contraction:
            structural_boosts["mw_contractions_apostrophe"] = 14
        if likely_comma_splice:
            structural_boosts["mw_run_on_comma_splice"] = 20
            structural_boosts["mw_punctuation_commas"] = 12

        scored_rules = []
        for rule in self.kb.rules:
            rid = rule.get("id", "")
            score = structural_boosts.get(rid, 0)
            target_words = rule.get("target_words", [])
            keywords = rule.get("keywords", [])

            # High weight for target word/phrase presence in student text
            for tw in target_words:
                tw_lower = tw.lower()
                if " " in tw_lower:
                    if tw_lower in text_lower:
                        score += 25
                elif tw_lower in words:
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

        scored_rules.sort(key=lambda x: x[0], reverse=True)
        return [rule for score, rule in scored_rules if score >= 8][:top_k]

    def retrieve_for_mistake(
        self,
        original: str,
        correction: str = "",
        category: str = "",
        explanation: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Find the most authoritative matching rule for a narrow word/phrase mistake snippet."""
        if not self.kb.rules:
            return None

        orig_clean = (original or "").strip().lower()
        corr_clean = (correction or "").strip().lower()
        exp_lower = (explanation or "").strip().lower()
        combined_text = f"{orig_clean} {corr_clean}".strip()
        tokens = set(re.findall(r"\b[a-zA-Z']+\b", combined_text))

        best_rule = None
        best_score = 0

        for rule in self.kb.rules:
            rid = rule.get("id", "")
            score = 0
            target_words = [tw.lower() for tw in rule.get("target_words", [])]

            # 1. Exact token match in target words
            for token in tokens:
                if token in target_words:
                    score += 25

            # 2. Multi-word target phrase match (e.g. "a lot", "urban city", "every day")
            for tw in target_words:
                if " " in tw and (tw in orig_clean or tw in corr_clean):
                    score += 30

            # Disambiguate overlapping target_word sets ('the' vs 'a/an', place vs time 'in/on/at')
            if rid == "mw_articles_a_vs_an" and "the" in tokens and not ({"a", "an"} & tokens):
                score = 0
            if rid == "mw_definite_article_the" and "the" in tokens and not ({"a", "an"} & tokens):
                score += 10
            if rid == "mw_prepositions_in_on_at_place" and ({"in", "on", "at"} & tokens):
                if any(h in exp_lower or h in combined_text for h in _PLACE_HINTS):
                    score += 12
            if rid == "mw_prepositions_in_on_at_time" and ({"in", "on", "at"} & tokens):
                if any(h in exp_lower or h in combined_text for h in _TIME_HINTS):
                    score += 12

            # 3. Keyword matches (word-boundary aware)
            for kw in rule.get("keywords", []):
                kw_lower = kw.lower()
                if " " in kw_lower:
                    if kw_lower in combined_text or (exp_lower and kw_lower in exp_lower):
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

        if best_score >= 20:
            return best_rule
        return None

    def retrieve_for_breakdown_item(
        self,
        original_part: str,
        suggested_part: str,
        part_of_speech: str,
        explanation: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Purposefully retrieve the exact Merriam-Webster rule for a single word,
        part-of-speech, or punctuation breakdown step."""
        if not self.kb.rules:
            return None

        clean_orig = _strip_placeholder(original_part)
        clean_sugg = _strip_placeholder(suggested_part)
        pos_lower = (part_of_speech or "").strip().lower()
        exp_lower = (explanation or "").strip().lower()
        combined = f"{clean_orig} {clean_sugg}".strip().lower()
        tokens = set(re.findall(r"\b[a-zA-Z']+\b", combined))

        # 1. High-precision structural & lexical checks
        if "redundan" in exp_lower or "pleonasm" in exp_lower or "urban city" in combined:
            return self.kb.get_by_id("mw_redundancy_pleonasm")

        if (clean_orig == "i" and clean_sugg == "I") or (
            clean_sugg == "I" and ("capital" in pos_lower or "capital" in exp_lower)
        ):
            return self.kb.get_by_id("mw_capitalization_pronoun_i")

        if {"everyday"} & tokens or "every day" in combined:
            return self.kb.get_by_id("mw_everyday_vs_every_day")

        if "alot" in tokens or "a lot" in combined:
            return self.kb.get_by_id("mw_spelling_a_lot")

        if {"its", "it's"} & tokens or (clean_orig.lower() == "it's" and "it is" in clean_sugg.lower()):
            return self.kb.get_by_id("mw_its_vs_its")

        if "contraction" in exp_lower or any(
            t in tokens for t in ("dont", "don't", "cant", "can't", "wont", "won't", "didnt", "didn't", "i'm")
        ):
            return self.kb.get_by_id("mw_contractions_apostrophe")

        # 2. Punctuation & Capitalization checks
        if "punctuation" in pos_lower or (not tokens and any(p in combined for p in ".!?,;:'\"")):
            if "comma splice" in exp_lower or "run-on" in exp_lower or ";" in combined:
                return self.kb.get_by_id("mw_run_on_comma_splice")
            if "," in combined or "comma" in exp_lower:
                if "interjection" in pos_lower or "interjection" in exp_lower:
                    return self.kb.get_by_id("mw_pos_interjection_register")
                return self.kb.get_by_id("mw_punctuation_commas")
            if "'" in combined or "apostrophe" in exp_lower:
                return self.kb.get_by_id("mw_contractions_apostrophe")
            return self.kb.get_by_id("mw_punctuation_end_marks_caps")

        if "capitalization" in pos_lower:
            if "i" in tokens:
                return self.kb.get_by_id("mw_capitalization_pronoun_i")
            return self.kb.get_by_id("mw_punctuation_end_marks_caps")

        # 3. Narrow phrase/word rule lookup when snippet is short
        if len(tokens) <= 4 and (clean_orig or clean_sugg):
            narrow = self.retrieve_for_mistake(
                original=clean_orig,
                correction=clean_sugg,
                explanation=explanation,
            )
            if narrow:
                # Guard against generic article/preposition match if POS explicitly says Verb/Noun/Adjective
                narrow_id = narrow.get("id", "")
                if narrow_id in ("mw_articles_a_vs_an", "mw_definite_article_the") and (
                    "article" not in pos_lower and "determiner" not in pos_lower and len(tokens) > 2
                ):
                    pass
                elif narrow_id == "mw_capitalization_pronoun_i" and not (
                    clean_orig == "i" or "capital" in pos_lower or "capital" in exp_lower
                ):
                    pass
                else:
                    return narrow

        # 4. Direct Part-of-Speech & Clause mapping to curated Merriam-Webster POS rules
        if "article" in pos_lower or "determiner" in pos_lower:
            if "the" in tokens and not ({"a", "an"} & tokens):
                return self.kb.get_by_id("mw_definite_article_the")
            return self.kb.get_by_id("mw_articles_a_vs_an")

        if "interjection" in pos_lower:
            return self.kb.get_by_id("mw_pos_interjection_register")

        if "conjunction" in pos_lower:
            return self.kb.get_by_id("mw_pos_conjunction_clauses")

        if "preposition" in pos_lower:
            if {"to", "for"} & tokens:
                return self.kb.get_by_id("mw_preposition_to_vs_for")
            if {"by", "with"} & tokens:
                return self.kb.get_by_id("mw_preposition_by_vs_with")
            if {"between", "among"} & tokens:
                return self.kb.get_by_id("mw_between_vs_among")
            if {"in", "on", "at"} & tokens:
                if any(h in exp_lower for h in _TIME_HINTS):
                    return self.kb.get_by_id("mw_prepositions_in_on_at_time")
                return self.kb.get_by_id("mw_prepositions_in_on_at_place")
            return self.kb.get_by_id("mw_pos_preposition_collocations")

        if "adverb" in pos_lower:
            return self.kb.get_by_id("mw_pos_adverb_usage")

        if "adjective" in pos_lower:
            return self.kb.get_by_id("mw_pos_adjective_precision")

        if "pronoun" in pos_lower:
            if {"who", "whom"} & tokens:
                return self.kb.get_by_id("mw_who_vs_whom")
            if clean_orig == "i" or ("capital" in exp_lower and "i" in tokens):
                return self.kb.get_by_id("mw_capitalization_pronoun_i")
            return self.kb.get_by_id("mw_pos_pronoun_clarity")

        if "verb" in pos_lower:
            if "fragment" in exp_lower and len(tokens) >= 2:
                return self.kb.get_by_id("mw_sentence_fragments")
            if "infinitive" in exp_lower and ({"to", "for"} & tokens):
                return self.kb.get_by_id("mw_preposition_to_vs_for")
            return self.kb.get_by_id("mw_pos_verb_agreement_tense")

        if "noun" in pos_lower:
            return self.kb.get_by_id("mw_pos_noun_usage")

        if "clause" in pos_lower or "phrase" in pos_lower:
            if "run-on" in exp_lower or "comma splice" in exp_lower:
                return self.kb.get_by_id("mw_run_on_comma_splice")
            if "conjunction" in exp_lower or (tokens & _SUBORDINATORS):
                return self.kb.get_by_id("mw_pos_conjunction_clauses")
            return self.kb.get_by_id("mw_sentence_fragments")

        return None

    def retrieve_for_sentence_mistake(
        self,
        original: str,
        correction: str,
        category: str = "",
        explanation: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Purposefully match a sentence-level revision by inspecting ONLY changed tokens,
        structural patterns, and category — never background unchanged function words."""
        if not self.kb.rules:
            return None

        orig_lower = (original or "").strip().lower()
        corr_lower = (correction or "").strip().lower()
        exp_lower = (explanation or "").strip().lower()
        cat_lower = (category or "").strip().lower()

        # Check explicit structural cues in explanation or text first
        if "fragment" in exp_lower or "missing subject" in exp_lower or "missing verb" in exp_lower:
            return self.kb.get_by_id("mw_sentence_fragments")
        if "run-on" in exp_lower or "comma splice" in exp_lower:
            return self.kb.get_by_id("mw_run_on_comma_splice")
        if "redundan" in exp_lower or "pleonasm" in exp_lower or "urban city" in orig_lower:
            return self.kb.get_by_id("mw_redundancy_pleonasm")

        # Diff tokens so unchanged background words ("the", "to", "a") do not false-match
        orig_tokens = re.findall(r"\b[a-zA-Z']+\b", orig_lower)
        corr_tokens = re.findall(r"\b[a-zA-Z']+\b", corr_lower)
        changed_tokens = set(orig_tokens) ^ set(corr_tokens)

        if changed_tokens:
            diff_snippet_orig = " ".join(t for t in orig_tokens if t in changed_tokens)
            diff_snippet_corr = " ".join(t for t in corr_tokens if t in changed_tokens)
            matched = self.retrieve_for_mistake(
                original=diff_snippet_orig,
                correction=diff_snippet_corr,
                category=category,
                explanation=explanation,
            )
            if matched:
                return matched

        if "," in corr_lower and "," not in orig_lower:
            return self.kb.get_by_id("mw_punctuation_commas")

        if (set(corr_tokens) | set(orig_tokens)) & _SUBORDINATORS:
            return self.kb.get_by_id("mw_pos_conjunction_clauses")

        # Category-aligned authoritative fallback from KB
        category_defaults = {
            "sentence structure": "mw_sentence_fragments",
            "clarity": "mw_pos_pronoun_clarity",
            "punctuation": "mw_punctuation_commas" if "," in corr_lower else "mw_punctuation_end_marks_caps",
            "word choice": "mw_pos_adjective_precision",
            "grammar": "mw_pos_verb_agreement_tense",
            "spelling": "mw_contractions_apostrophe",
        }
        default_id = category_defaults.get(cat_lower, "mw_sentence_fragments")
        return self.kb.get_by_id(default_id)

    def build_rag_prompt_section(self, rules: List[Dict[str, Any]]) -> str:
        """Format retrieved authoritative rules + compact KB catalog for structured RAG tagging."""
        if not self.kb.rules:
            return ""

        selected = rules or []
        selected_ids = {r.get("id") for r in selected if r.get("id")}

        lines = [
            "AUTHORITATIVE REFERENCE RULES (Merriam-Webster):",
            "Use these verified rules to ground your explanations, and set `rule_id` on matching "
            "sentence items and word/punctuation breakdown steps:",
        ]

        for rule in selected:
            lines.append(
                f"- [{rule.get('id')}] {rule.get('title')}: {rule.get('rule_summary')}"
            )

        remaining = [r for r in self.kb.rules if r.get("id") not in selected_ids]
        if remaining:
            catalog = "; ".join(f"[{r.get('id')}] {r.get('title')}" for r in remaining)
            lines.append(f"\nADDITIONAL RULE IDs FOR WORD & POS BREAKDOWN: {catalog}")

        return "\n".join(lines)

    def ground_breakdown_item(self, item: Any) -> WordBreakdownItem:
        """Enrich a single word/POS/punctuation breakdown step with Merriam-Webster RAG evidence."""
        orig_part = getattr(item, "original_part", "") if not isinstance(item, dict) else item.get("original_part", "")
        sugg_part = getattr(item, "suggested_part", "") if not isinstance(item, dict) else item.get("suggested_part", "")
        pos = getattr(item, "part_of_speech", "") if not isinstance(item, dict) else item.get("part_of_speech", "")
        exp = getattr(item, "explanation", "") if not isinstance(item, dict) else item.get("explanation", "")
        rule_id = getattr(item, "rule_id", None) if not isinstance(item, dict) else item.get("rule_id")
        rule_name = getattr(item, "rule_name", None) if not isinstance(item, dict) else item.get("rule_name")
        source_name = getattr(item, "source_name", None) if not isinstance(item, dict) else item.get("source_name")
        source_url = getattr(item, "source_url", None) if not isinstance(item, dict) else item.get("source_url")
        proof = getattr(item, "proof", None) if not isinstance(item, dict) else item.get("proof")

        orig_part = orig_part or "(missing)"
        sugg_part = sugg_part or "(removed)"
        pos = pos or "Grammar"
        exp = exp or ""

        if source_url and proof and rule_name and source_name:
            return WordBreakdownItem(
                original_part=orig_part,
                suggested_part=sugg_part,
                part_of_speech=pos,
                explanation=exp,
                rule_id=rule_id,
                rule_name=rule_name,
                source_name=source_name,
                source_url=source_url,
                proof=proof,
            )

        matched_rule = self.kb.get_by_id(rule_id) if rule_id else None
        if not matched_rule:
            matched_rule = self.retrieve_for_breakdown_item(
                original_part=orig_part,
                suggested_part=sugg_part,
                part_of_speech=pos,
                explanation=exp,
            )

        if matched_rule:
            return WordBreakdownItem(
                original_part=orig_part,
                suggested_part=sugg_part,
                part_of_speech=pos,
                explanation=exp,
                rule_id=matched_rule.get("id"),
                rule_name=rule_name or matched_rule.get("title"),
                source_name=source_name or matched_rule.get("source_name", "Merriam-Webster"),
                source_url=source_url or matched_rule.get("source_url"),
                proof=proof or matched_rule.get("proof"),
            )

        # Dictionary fallback for specific vocabulary / spelling words
        clean_target = _strip_placeholder(sugg_part) or _strip_placeholder(orig_part)
        tokens = re.findall(r"\b[a-zA-Z]+\b", clean_target)
        clean_word = tokens[0].lower() if tokens else "grammar"
        dict_url = f"https://www.merriam-webster.com/dictionary/{clean_word}"

        return WordBreakdownItem(
            original_part=orig_part,
            suggested_part=sugg_part,
            part_of_speech=pos,
            explanation=exp,
            rule_id=rule_id,
            rule_name=rule_name or f"{pos} Usage: '{clean_target or clean_word}'",
            source_name=source_name or "Merriam-Webster",
            source_url=source_url or dict_url,
            proof=proof or (
                f"Merriam-Webster Dictionary & Usage ({pos}): '{clean_target or clean_word}' — {exp}"
            ),
        )

    def ground_mistake(self, mistake: Any) -> MistakeItem:
        """Enrich a MistakeItem AND all of its word/POS/punctuation breakdown steps
        with authoritative Merriam-Webster proof and links."""
        orig = getattr(mistake, "original", "") or ""
        corr = getattr(mistake, "correction", "") or ""
        cat = getattr(mistake, "category", "") or "Grammar"
        exp = getattr(mistake, "explanation", "") or ""
        rule_id = getattr(mistake, "rule_id", None)
        rule_name = getattr(mistake, "rule_name", None)
        source_name = getattr(mistake, "source_name", None)
        source_url = getattr(mistake, "source_url", None)
        proof = getattr(mistake, "proof", None)
        raw_breakdown = getattr(mistake, "breakdown", None) or []

        grounded_breakdown = [self.ground_breakdown_item(b) for b in raw_breakdown]

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
                breakdown=grounded_breakdown,
            )

        # 1. Direct lookup by rule_id if provided by the model
        matched_rule = self.kb.get_by_id(rule_id) if rule_id else None

        is_sentence_level = len(re.findall(r"\b\w+\b", orig)) > SENTENCE_LEVEL_MIN_WORDS

        # 2. Phrase-level heuristic lookup
        if not matched_rule and not is_sentence_level:
            matched_rule = self.retrieve_for_mistake(
                original=orig,
                correction=corr,
                category=cat,
                explanation=exp,
            )

        # 3. Sentence-level purposeful RAG grounding: synthesize from breakdown rules or diffed tokens
        if not matched_rule and is_sentence_level:
            # Prefer a KB rule matched in the breakdown (prioritizing structural/category matches)
            for b in grounded_breakdown:
                if b.rule_id and self.kb.get_by_id(b.rule_id):
                    candidate = self.kb.get_by_id(b.rule_id)
                    if candidate and candidate.get("category", "").lower() == cat.lower():
                        matched_rule = candidate
                        break
                    if not matched_rule:
                        matched_rule = candidate

            if not matched_rule:
                matched_rule = self.retrieve_for_sentence_mistake(
                    original=orig,
                    correction=corr,
                    category=cat,
                    explanation=exp,
                )

        if matched_rule:
            resolved_id = matched_rule.get("id")
            resolved_name = rule_name or matched_rule.get("title")
            resolved_source = source_name or matched_rule.get("source_name", "Merriam-Webster")
            resolved_url = source_url or matched_rule.get("source_url") or MW_GRAMMAR_HUB_URL
            resolved_proof = proof or matched_rule.get("proof")

            return MistakeItem(
                original=orig,
                correction=corr,
                category=cat,
                explanation=exp,
                rule_id=resolved_id,
                rule_name=resolved_name,
                source_name=resolved_source,
                source_url=resolved_url,
                proof=resolved_proof,
                breakdown=grounded_breakdown,
            )

        # 4. Fallback dictionary grounding for short vocabulary/spelling items
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
            breakdown=grounded_breakdown,
        )

    def enrich_review_response(
        self,
        response: Any,
        token_usage: Optional[Any] = None,
    ) -> ReviewResponse:
        """Enrich all mistakes and word breakdowns in an AI or mock response with verified proof and links."""
        if not response:
            return ReviewResponse(
                mistakes=[], summary="", overall_feedback="", scores=None, token_usage=token_usage
            )

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
