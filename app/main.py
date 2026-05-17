import json
import hashlib
import re
from pathlib import Path

from fastapi import FastAPI
from dotenv import load_dotenv

# =====================================================================
# ENV LOADING & HARDENING BLOCK
# Forces absolute resolution of the directory path to ensure your 
# GEMINI_API_KEY is correctly injected into the process context on Windows.
# =====================================================================
base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / ".env"

print(f"\n--- [ENV DIAGNOSTIC] Checking path: {env_path.absolute()} ---")
print(f"--- [ENV DIAGNOSTIC] Does file exist? {env_path.exists()} ---\n")

load_dotenv(dotenv_path=env_path, override=True)

# Verify keys loaded correctly
import os
_keys_raw = os.environ.get("GEMINI_API_KEYS", "NOT FOUND")
_key_count = len([k for k in _keys_raw.split(",") if k.strip()]) if _keys_raw != "NOT FOUND" else 0
print(f"--- [KEY DIAGNOSTIC] GEMINI_API_KEYS found: {_keys_raw != 'NOT FOUND'}, key count: {_key_count} ---\n")

# Delayed imports to guarantee environment context is populated prior to service compilation
from app.routes import chat


def _infer_test_type(name: str, description: str) -> str:
    s = (name + " " + description).lower()
    
    # Cognitive and knowledge items map to "K" per assignment specification guidelines
    cognitive_kw = ["ability", "aptitude", "cognitive", "reasoning", "gsa", "critical thinking"]
    personality_kw = ["personality", "behavior", "opq", "style", "trait", "culture"]
    coding_kw = ["coding", "java", "python", "sql", "technical", "skills", "engineering"]
    situational_kw = ["situational", "judgment", "sjt", "scenario", "simulation"]

    for kw in cognitive_kw:
        if kw in s:
            return "K"
    for kw in personality_kw:
        if kw in s:
            return "P"
    for kw in coding_kw:
        if kw in s:
            return "K"
    for kw in situational_kw:
        if kw in s:
            return "S"
            
    return "K"  # Default safe compliance fallback identifier


def _tokenize_name(name: str) -> list:
    tokens = re.findall(r"\b\w+\b", name.lower())
    return [t for t in tokens if t]


def _normalize_item(item: dict) -> dict:
    name = (item.get("name") or "").strip()
    description = item.get("description") or ""

    # Defensively map crawled links to target URL structures
    url = item.get("link") or item.get("url") or "https://www.shl.com/solutions/products/product-catalog/"

    technical_categories = item.get("technical_categories")
    if technical_categories is None:
        technical_categories = item.get("keys") or []
    # Ensure list of strings
    technical_categories = [str(x) for x in technical_categories] if technical_categories else []

    # Enforce clear literal classification matching
    test_type = item.get("test_type") or _infer_test_type(name, description)
    test_type = str(test_type).upper().strip()
    if test_type not in ["K", "P", "S", "C"]:
        test_type = "K"

    # HARDENING LAYER: Build explicit alias lists containing common shortened acronyms
    aliases = item.get("aliases") or []
    aliases = [str(a).lower().strip() for a in aliases]
    
    # Extract any alphanumeric short codes present in parentheses like (OPQ32r)
    for match in re.findall(r"\((.*?)\)", name.lower()):
        aliases.extend(match.split())
    
    # Always include baseline name tokens for backup alignment
    aliases.extend(_tokenize_name(name))
    aliases = list(set([a for a in aliases if a]))

    entity_id = item.get("entity_id")
    if entity_id is None:
        # Deterministic md5 of name
        entity_id = hashlib.md5(name.encode("utf-8") if name else b"").hexdigest()
    else:
        entity_id = str(entity_id)

    job_levels = item.get("job_levels") or []

    normalized = {
        "entity_id": entity_id,
        "name": name,
        "url": url,
        "test_type": test_type,
        "job_levels": job_levels,
        "technical_categories": technical_categories,
        "description": description,
        "aliases": aliases,
    }
    # Preserve other original fields as optional metadata
    normalized.update({k: v for k, v in item.items() if k not in normalized})
    return normalized


def create_app() -> FastAPI:
    app = FastAPI(title="SHL Conversational Assessment Recommender Engine")

    @app.on_event("startup")
    def load_catalog():
        base = Path(__file__).resolve().parents[1]
        catalog_path = base / "data" / "catalog.json"
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                raw = json.load(f, strict=False)
        else:
            raw = []

        normalized = []
        for item in raw:
            try:
                normalized.append(_normalize_item(item))
            except Exception:
                # Skip malformed entries safely to protect pipeline boot
                continue

        app.state.catalog = normalized

    @app.get("/health")
    def health():
        """Mandatory endpoint confirming API readiness status to the evaluation replay harness."""
        return {"status": "ok"}

    # Include the chat router
    app.include_router(chat.router)

    return app


app = create_app()