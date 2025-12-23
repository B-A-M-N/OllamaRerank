# rerank_stack/src/rerank_service/policy.py
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

print("--- policy.py loaded (DEBUG CONFIRMATION) ---") # Temporary debug line



@dataclass
class Policy:
    name: str
    scale: List[int]
    synonyms: Dict[str, List[str]] = field(default_factory=dict)
    topic_anchors: List[str] = field(default_factory=list)
    boost_anchors: List[str] = field(default_factory=list)
    shower_context_anchors: List[str] = field(default_factory=list)
    tier_a_keywords: List[str] = field(default_factory=list)
    tier_b_keywords: List[str] = field(default_factory=list)
    intent_keywords: List[str] = field(default_factory=list)
    ambiguous_triggers: List[str] = field(default_factory=list)
    ambiguous_max_score: Optional[int] = None
    rubric: Optional[str] = None
    domain_keywords: List[str] = field(default_factory=list)
    item_gating_rules: Dict[str, Any] = field(default_factory=dict)
    caps: List[Dict[str, Any]] = field(default_factory=list)
    boosts: List[Dict[str, Any]] = field(default_factory=list)
    few_shot: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def synonym_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for canonical, variants in self.synonyms.items():
            mapping[canonical.lower()] = canonical.lower()
            for variant in variants:
                mapping[variant.lower()] = canonical.lower()
        return mapping


@dataclass
class PolicyContext:
    policy: Policy
    synonym_map: Dict[str, str]
    topic_anchors: Set[str]
    boost_anchors: Set[str]
    tier_a_keywords: Set[str]
    tier_b_keywords: Set[str]
    intent_keywords: Set[str]
    ambiguous_triggers: Set[str]
    ambiguous_max_score: Optional[int]
    domain_keywords: Set[str]
    shower_context_anchors: Set[str]

    # NEW: canonical_item -> set(aliases)
    item_aliases: Dict[str, Set[str]]

    def to_bucket_index(self, legacy_or_bucket: int) -> int:
        """
        Accept either:
          - legacy score in {0,3,7,10}
          - bucket index in {0,1,2,3}
        Return bucket index 0..3.
        """
        if legacy_or_bucket in (0, 1, 2, 3):
            return int(legacy_or_bucket)
        if legacy_or_bucket == 0:
            return 0
        if legacy_or_bucket == 3:
            return 1
        if legacy_or_bucket == 7:
            return 2
        if legacy_or_bucket == 10:
            return 3
        # fallback: clamp-ish
        if legacy_or_bucket < 3:
            return 0
        if legacy_or_bucket < 7:
            return 1
        if legacy_or_bucket < 10:
            return 2
        return 3

    def _matches_any(self, text: str, terms: Set[str]) -> bool:
        if not terms:
            return False
        t = text or ""
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", t, re.IGNORECASE):
                return True
        return False

    def _find_items_in_text(self, text: str) -> List[str]:
        hits: List[str] = []
        for item, aliases in self.item_aliases.items():
            if self._matches_any(text, aliases):
                hits.append(item)
        return hits

    def _infer_query_item(self, query: str) -> Optional[str]:
        # Prefer explicit match from item aliases
        hits = self._find_items_in_text(query)
        print(f"DEBUG infer_query_item hits:", hits, "query:", query)
        if len(hits) == 1:
            return hits[0]
        # If multiple items appear, we treat as ambiguous (None)
        return None

    def _domain_match(self, text: str) -> bool:
        # Soft domain check: require at least one domain keyword hit.
        if not self.domain_keywords:
            return True
        return bool(self._matches_any(text, self.domain_keywords))

    # --- Domain inference helpers ---
    def _domain_hits(self, text: str) -> Dict[str, int]:
        hits: Dict[str, int] = {}
        t = text or ""
        for rule in DOMAIN_RULES:
            count = sum(1 for p in rule.patterns if p.search(t))
            if count > 0:
                hits[rule.name] = hits.get(rule.name, 0) + count
        return hits

    def _choose_domain(self, hit_map: Dict[str, int]) -> Optional[str]:
        if not hit_map:
            return None
        best = max(hit_map.items(), key=lambda kv: kv[1])
        return best[0] if best[1] > 0 else None

    def compute_identity_facts(self, query: str, doc: str) -> Dict[str, Any]:
        q_item = self._infer_query_item(query)
        doc_items = self._find_items_in_text(doc)
        if not doc_items:
            same_item = None
        else:
            same_item = bool(q_item) and (q_item in doc_items)

        # Domain detection via anchor hits
        query_domain_hits = self._domain_hits(query)
        doc_domain_hits = self._domain_hits(doc)
        query_domain = self._choose_domain(query_domain_hits)
        doc_domain = self._choose_domain(doc_domain_hits)
        same_domain = bool(query_domain) and (query_domain == doc_domain)
        domain_match_doc = bool(doc_domain_hits)
        domain_match_query = bool(query_domain_hits)

        # Debug block to surface why facts might be empty/broken
        print(
            "[FACTS_DEBUG]",
            "policy_name=", getattr(self.policy, "name", None),
            "num_items=", len(self.item_aliases or {}),
            "num_domain_keywords=", len(self.domain_keywords or []),
            "query_item=", q_item,
            "doc_items=", doc_items,
            "query_domain=", query_domain,
            "doc_domain=", doc_domain,
            "same_domain=", same_domain,
            "query_domain_hits=", query_domain_hits,
            "doc_domain_hits=", doc_domain_hits,
            "same_item=", same_item,
            "query_norm=", repr((query or "")[:80]),
            "doc_norm=", repr((doc or "")[:80]),
        )

        return {
            "query_item": q_item,
            "doc_items": doc_items,
            "same_item": same_item,
            # legacy field kept for compatibility; now represents "same domain"
            "domain_match": same_domain,
            "domain_match_doc": domain_match_doc,
            "domain_match_query": domain_match_query,
            "query_domain": query_domain,
            "doc_domain": doc_domain,
            "query_domain_hits": query_domain_hits,
            "doc_domain_hits": doc_domain_hits,
            "same_domain": same_domain,
        }


@dataclass(frozen=True)
class DomainRule:
    name: str
    patterns: List[re.Pattern]
    negative_patterns: List[re.Pattern]


def _compile_words(words: List[str]) -> List[re.Pattern]:
    return [re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in words if word]


DOMAIN_RULES: List[DomainRule] = [
    DomainRule(
        name="automotive",
        patterns=_compile_words(["car", "auto", "automotive", "engine", "brake", "oil", "spark", "civic", "torque", "lug"]),
        negative_patterns=[],
    ),
    DomainRule(
        name="bicycle",
        patterns=_compile_words(["bicycle", "bike", "chain", "chains", "tire", "wheel", "derailleur", "cassette", "pump"]),
        negative_patterns=[],
    ),
    DomainRule(
        name="plumbing",
        patterns=_compile_words(["plumbing", "plumber", "shower", "faucet", "cartridge", "valve", "pipe", "tub", "bath", "drain"]),
        negative_patterns=[],
    ),
    DomainRule(
        name="plumbing",
        patterns=_compile_words(["water heater", "heater"]),
        negative_patterns=[],
    ),
]


PROCEDURAL_INTENT_KEYWORDS = {
    "how", "fix", "replace", "repair", "install", "remove",
    "troubleshoot", "steps", "guide", "tutorial", "change"
}

MAINTENANCE_INTENT_KEYWORDS = {
    "maintenance", "maintain", "service", "servicing", "care", "upkeep", "routine"
}

def build_policy_context(policy: Policy) -> PolicyContext:
    # Build canonical_item -> aliases set from item_gating_rules + synonyms
    item_aliases: Dict[str, Set[str]] = {}

    # 1) From item_gating_rules
    for item, rules in (policy.item_gating_rules or {}).items():
        item_l = str(item).lower()
        item_aliases.setdefault(item_l, set()).add(item_l)
        if isinstance(rules, dict):
            must = rules.get("must_match_any", []) or []
            for a in must:
                item_aliases[item_l].add(str(a).lower())

    # 2) From synonyms (optional bonus)
    # If you have synonyms like faucet->tap, treat those as aliases for identity matching too.
    for canonical, variants in (policy.synonyms or {}).items():
        c = str(canonical).lower()
        if c in item_aliases:
            for v in variants:
                item_aliases[c].add(str(v).lower())

    return PolicyContext(
        policy=policy,
        synonym_map=policy.synonym_map,
        topic_anchors=set(policy.topic_anchors),
        boost_anchors=set(policy.boost_anchors),
        tier_a_keywords=set(policy.tier_a_keywords or policy.topic_anchors),
        tier_b_keywords=set(policy.tier_b_keywords),
        intent_keywords=PROCEDURAL_INTENT_KEYWORDS | MAINTENANCE_INTENT_KEYWORDS,
        ambiguous_triggers=set(policy.ambiguous_triggers),
        ambiguous_max_score=policy.ambiguous_max_score,
        domain_keywords={str(k).lower() for k in policy.domain_keywords},
        shower_context_anchors=set(policy.shower_context_anchors),
        item_aliases=item_aliases,
    )


def load_policy(path: Path | str) -> Policy:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {config_path}")
    with open(config_path, "r") as f:
        if config_path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    raw_synonyms = data.get("synonyms", {})
    normalized_synonyms = _normalize_synonyms(raw_synonyms)
    raw_item_gating_rules = data.get("item_gating_rules", {})
    normalized_item_gating_rules = _normalize_item_gating_rules(raw_item_gating_rules)
    # Fallback: some configs still use object_groups; treat them as item aliases.
    if not normalized_item_gating_rules:
        raw_object_groups = data.get("object_groups", {})
        if isinstance(raw_object_groups, dict):
            iterable = raw_object_groups.items()
        elif isinstance(raw_object_groups, list):
            # list of dicts: [{"car": [...]}, {"bike": [...]}]
            iterable = []
            for entry in raw_object_groups:
                if isinstance(entry, dict):
                    iterable.extend(entry.items())
        else:
            iterable = []
        for item, aliases in iterable:
            normalized_item_gating_rules[str(item).lower()] = {
                "must_match_any": _normalize_value(aliases)
            }
    return Policy(
        name=data.get("name", "unnamed"),
        scale=data.get("scale", [0, 3, 7, 10]),
        synonyms=normalized_synonyms,
        topic_anchors=data.get("topic_anchors", []),
        boost_anchors=data.get("boost_anchors", []),
        shower_context_anchors=data.get("shower_context_anchors", []),
        tier_a_keywords=data.get("tier_a_keywords", []),
        tier_b_keywords=data.get("tier_b_keywords", []),
        caps=data.get("caps", []),
        boosts=data.get("boosts", []),
        intent_keywords=data.get("intent_keywords", []),
        ambiguous_triggers=data.get("ambiguous_triggers", []),
        ambiguous_max_score=data.get("ambiguous_max_score"),
        rubric=data.get("rubric"),
        domain_keywords=data.get("domain_keywords", []),
        item_gating_rules=normalized_item_gating_rules,
        few_shot=data.get("few_shot", []),
    )


def _policy_files(directory: Path) -> List[Path]:
    files = []
    for ext in ("*.yaml", "*.yml", "*.json"):
        files.extend(sorted(directory.glob(ext)))
    return files


def _normalize_value(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).lower() for v in value]
    return [str(value).lower()]


def _normalize_synonyms(raw_synonyms: Any) -> Dict[str, List[str]]:
    if isinstance(raw_synonyms, dict):
        return {
            str(key).lower(): _normalize_value(value) for key, value in raw_synonyms.items()
        }
    if isinstance(raw_synonyms, list):
        normalized: Dict[str, List[str]] = {}
        for entry in raw_synonyms:
            if isinstance(entry, dict):
                for key, value in entry.items():
                    normalized[str(key).lower()] = _normalize_value(value)
        return normalized
    return {}


def _normalize_item_gating_rules(raw_rules: Any) -> Dict[str, Any]:
    """
    Accept either:
      - dict: {item: {must_match_any: [...], neg_if_doc_has: [...]}}
      - list of dicts with nested dict/list (some configs use this shape)
    Normalize keys to lowercase and values to lowercase lists.
    """
    normalized_rules: Dict[str, Any] = {}

    def _normalize_rule_dict(item: str, rules_dict: Dict[str, Any]):
        normalized_rules[str(item).lower()] = {
            str(k).lower(): _normalize_value(v) for k, v in (rules_dict or {}).items()
        }

    if isinstance(raw_rules, dict):
        for item, rules in raw_rules.items():
            if isinstance(rules, dict):
                _normalize_rule_dict(item, rules)
        return normalized_rules

    if isinstance(raw_rules, list):
        for entry in raw_rules:
            if not isinstance(entry, dict):
                continue
            for item, rules in entry.items():
                # some configs encode rules as a list of dicts
                if isinstance(rules, list):
                    merged: Dict[str, Any] = {}
                    for r in rules:
                        if isinstance(r, dict):
                            merged.update(r)
                    _normalize_rule_dict(item, merged)
                elif isinstance(rules, dict):
                    _normalize_rule_dict(item, rules)
        return normalized_rules

    return normalized_rules


class PolicyManager:
    def infer_domain(self, query: str) -> Optional[str]:
        best_domain: Optional[str] = None
        best_score = 0
        for rule in DOMAIN_RULES:
            if any(p.search(query) for p in rule.negative_patterns):
                continue
            score = sum(1 for p in rule.patterns if p.search(query))
            if score > best_score:
                best_score = score
                best_domain = rule.name
        return best_domain if best_score > 0 else None

    def __init__(self, config_dir: Path | None = None):
        base = Path(__file__).resolve().parents[2]
        self.config_dir = config_dir or base / "configs"
        self.policies: Dict[str, Policy] = {}
        self.default_policy_id: Optional[str] = None
        self._config_snapshot: Dict[Path, float] = {}
        self.load_policies()

    def load_policies(self):
        file_stats = self._policy_file_stats()
        if not file_stats:
            raise RuntimeError(f"No policy configs found in {self.config_dir}")
        self.policies.clear()
        self.default_policy_id = None
        for config_path in sorted(file_stats):
            policy = load_policy(config_path)
            self.policies[policy.name] = policy
            if not self.default_policy_id:
                self.default_policy_id = policy.name
        self._config_snapshot = file_stats

    def select_policy(self, policy_id: str | None = None, query: str | None = None) -> Policy:
        self.ensure_policies_current()
        if policy_id and policy_id in self.policies:
            return self.policies[policy_id]
        if query:
            domain = self.infer_domain(query)
            if domain:
                candidate = f"howto_{domain}_v1"
                if candidate in self.policies:
                    return self.policies[candidate]
        default_id = self.default_policy_id or next(iter(self.policies))
        return self.policies[default_id]

    def _policy_file_stats(self) -> Dict[Path, float]:
        stats: Dict[Path, float] = {}
        for config_path in _policy_files(self.config_dir):
            try:
                stats[config_path] = config_path.stat().st_mtime
            except OSError:
                continue
        return stats

    def ensure_policies_current(self) -> None:
        current_stats = self._policy_file_stats()
        if current_stats != self._config_snapshot:
            self.load_policies()


policy_manager = PolicyManager()
