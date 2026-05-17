import json
import hashlib
import re
from pathlib import Path

from fastapi import FastAPI

from app.routes import chat


def _infer_test_type(name: str, description: str) -> str:
    s = (name + " " + description).lower()
    # cognitive indicators -> C
    cognitive_kw = ["ability", "aptitude", "cognitive", "reasoning", "gsa"]
    personality_kw = ["personality", "behavior", "opq", "style"]
    coding_kw = ["coding", "java", "python", "sql", "technical", "skills"]

    for kw in cognitive_kw:
        if kw in s:
            return "C"
    for kw in personality_kw:
        if kw in s:
            return "P"
    for kw in coding_kw:
        if kw in s:
            return "K"
    return "C"


def _tokenize_name(name: str) -> list:
    tokens = re.findall(r"\b\w+\b", name.lower())
    return [t for t in tokens if t]


def _normalize_item(item: dict) -> dict:
    name = (item.get("name") or "").strip()
    description = item.get("description") or ""

    url = item.get("link") or item.get("url") or ""

    technical_categories = item.get("technical_categories")
    if technical_categories is None:
        technical_categories = item.get("keys") or []
    # ensure list of strings
    technical_categories = [str(x) for x in technical_categories] if technical_categories else []

    test_type = item.get("test_type") or _infer_test_type(name, description)

    aliases = item.get("aliases")
    if not aliases:
        aliases = _tokenize_name(name)
    aliases = [str(a) for a in aliases]

    entity_id = item.get("entity_id")
    if entity_id is None:
        # deterministic md5 of name
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
    # preserve other original fields as optional metadata
    normalized.update({k: v for k, v in item.items() if k not in normalized})
    return normalized


def create_app() -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def load_catalog():
        base = Path(__file__).resolve().parents[1]
        catalog_path = base / "data" / "catalog.json"
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                raw = json.load(f, strict = False)
        else:
            raw = []

        normalized = []
        for item in raw:
            try:
                normalized.append(_normalize_item(item))
            except Exception:
                # skip malformed entries
                continue

        app.state.catalog = normalized

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # Include the chat router
    app.include_router(chat.router)

    return app


app = create_app()
