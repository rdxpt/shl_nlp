import math
import re
from collections import Counter
from typing import Dict, List

STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "is", "at", "by"
}

# Tightened, highly directed expansion map to prevent keyword dilution
# Update ONLY this block at the top of app/retrieval/search.py
SYNONYM_MAP = {
    "software engineer": ["developer", "coding", "programming", "software"],
    "graduate": ["entry-level", "junior", "aptitude"],
    "backend": ["java", "spring", "python", "sql", "api", "database"],
    "frontend": ["javascript", "react", "html", "css", "web"],
    "data scientist": ["python", "analytics", "statistics", "machine"],
    # 🚀 THE MISSING BINDINGS: Connects abstract intents to real title keywords
    "database development": ["sql", "database", "server", "oracle", "query"],
    "java backend systems": ["java", "spring", "backend", "mvc", "api"]
}


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _expand_query_string(query: str) -> str:
    query_lower = query.lower()
    expanded_tokens = _tokenize(query_lower)
    
    for trigger, synonyms in SYNONYM_MAP.items():
        if trigger in query_lower:
            expanded_tokens.extend(synonyms)
            
    return " ".join(set(expanded_tokens))


def _cosine_similarity(counter_a: Counter, counter_b: Counter) -> float:
    if not counter_a or not counter_b:
        return 0.0
    dot = 0
    for k, v in counter_a.items():
        dot += v * counter_b.get(k, 0)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_enterprise_bundle(record: Dict) -> bool:
    text_fields = []
    for key in ("name", "description"):
        v = record.get(key) or ""
        text_fields.append(str(v).lower())
    cats = record.get("technical_categories") or []
    text_fields.extend([str(x).lower() for x in cats if x])
    aliases = record.get("aliases") or []
    text_fields.extend([str(x).lower() for x in aliases if x])

    joined = " ".join(text_fields)
    enterprise_indicators = [
        "bundle", "enterprise", "pack", "package", "suite", "global",
        "organizational", "corporate", "hiring package", "talent suite",
        "enterprise solution", "pre-packaged"
    ]
    for kw in enterprise_indicators:
        if kw in joined:
            return True
    return False


def run_hybrid_rrf_search(query_string: str, catalog: List[dict], top_k: int = 10) -> List[dict]:
    if not query_string or not catalog:
        return []

    expanded_query = _expand_query_string(query_string)
    query_tokens = _tokenize(expanded_query)
    query_counter = Counter(query_tokens)

    # 1) Filter out enterprise bundles
    candidates = []
    for idx, rec in enumerate(catalog):
        if _is_enterprise_bundle(rec):
            continue
        candidates.append((idx, rec))

    if not candidates:
        return []

    # 2) Optimized Weighted Lexical Ranker (Stable Multipliers)
    lexical_scores = []
    for idx, rec in candidates:
        name_tokens = _tokenize(rec.get("name") or "")
        cats = [str(t).lower() for t in (rec.get("technical_categories") or []) if isinstance(t, str)]
        aliases = _tokenize(" ".join([str(a) for a in (rec.get("aliases") or []) if a]))
        
        score = 0.0
        for t, q_count in query_counter.items():
            # 🚀 FIXED: Use a flat token multiplier instead of dividing by title length
            if t in name_tokens:
                score += 4.0 * name_tokens.count(t)
            if t in cats:
                score += 2.0
            if t in aliases:
                score += 1.5
                
        lexical_scores.append((idx, score))
    lexical_scores.sort(key=lambda x: (-x[1], x[0]))
    lexical_rank = {}
    for rank, (idx, _) in enumerate(lexical_scores, start=1):
        lexical_rank[idx] = rank

    # 3) Semantic Ranker (Cosine Similarity)
    semantic_scores = []
    for idx, rec in candidates:
        desc = rec.get("description") or ""
        desc_tokens = _tokenize(desc)
        desc_counter = Counter(desc_tokens)
        sim = _cosine_similarity(query_counter, desc_counter)
        semantic_scores.append((idx, sim))

    semantic_scores.sort(key=lambda x: (-x[1], x[0]))
    semantic_rank = {}
    for rank, (idx, _) in enumerate(semantic_scores, start=1):
        semantic_rank[idx] = rank

    # 4) RRF Combination Loop
    results = []
    for idx, rec in candidates:
        l_rank = lexical_rank.get(idx, len(candidates) + 1)
        s_rank = semantic_rank.get(idx, len(candidates) + 1)
        
        # Reciprocal Rank Fusion formula
        rrf_score = 1.0 / (60.0 + float(l_rank)) + 1.0 / (60.0 + float(s_rank))
        results.append((rrf_score, idx, rec))

    results.sort(key=lambda x: (-x[0], x[1]))

    top = []
    for score, idx, rec in results[:top_k]:
        out = rec.copy()
        out["rrf_score"] = score
        top.append(out)

    return top