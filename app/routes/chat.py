from pathlib import Path
from typing import List

from fastapi import APIRouter, Request

from app.core.api_models import ChatRequest, ChatResponse, RecommendationItem, Message
from app.core.state_models import FeatureTracker
from app.services.parser_service import extract_system_state
from app.services.synthesis_service import synthesize

from app.retrieval.search import run_hybrid_rrf_search

router = APIRouter()


def _get_catalog_from_app(request: Request) -> List[dict]:
    return getattr(request.app.state, "catalog", [])


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, request: Request):
    messages: List[Message] = payload.messages
    current_turn = len(messages)

    # Reconstruct state from full transcript
    state: FeatureTracker = extract_system_state(messages)

    # Enforce hard turn cap rule: if current_turn >= 6, force context_complete = True
    if current_turn >= 6:
        state.context_complete = True

    # Gate 1: Safety firewall
    if state.is_jailbreak or state.out_of_scope:
        reply = synthesize("refusal", state, messages, [])
        # FIX: End the conversation gracefully on security refusal
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=True)

    # Gate 2: Comparison mode
    if state.wants_comparison:
        matched_records = []
        catalog = _get_catalog_from_app(request)
        for target in state.comparison_targets:
            res = run_hybrid_rrf_search(target, catalog, top_k=1)
            if res:
                matched_records.append(res[0])
        reply = synthesize("comparison", state, messages, matched_records)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # Gate 3: Recommendation flow & safety cap buffer
    if state.context_complete or current_turn >= 6:
        # Build query from slots
        parts = []
        if state.target_role:
            parts.append(state.target_role)
        if state.seniority_level:
            parts.append(state.seniority_level)
        
        query = " ".join(parts).strip()

        # 🚀 FALLBACK: If slots are missing due to a hard turn cap override, 
        # extract keywords from historical messages to ensure a valid search query
        if not query and messages:
            # Look backwards to find the last substantive keywords from the user
            for m in reversed(messages):
                if m.role == "user" and m.content:
                    # Filter out purely conversational noise if possible, or use the message
                    query = m.content
                    break

        # Run hybrid search to collect top unique matches
        catalog = _get_catalog_from_app(request)
        results = run_hybrid_rrf_search(query, catalog, top_k=10)

        # Build recommendations JSON using only verified catalog fields
        seen = set()
        recommendations = []
        matched_records = []
        for r in results:
            eid = r.get("entity_id") or r.get("name")
            if eid in seen:
                continue
            seen.add(eid)
            matched_records.append(r)
            recommendations.append(
                {
                    "name": r.get("name", ""),
                    "url": r.get("url", ""),
                    "test_type": r.get("test_type", ""),
                }
            )

        # FIX: If we successfully deliver recommendations OR we hit the turn cap, end the conversation
        end_of_conversation = True
        
        # Explicitly pass "recommendation" mode to bypass conversational confusion over missing slots
        reply = synthesize("recommendation", state, messages, matched_records)

        return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=end_of_conversation)

    # Else: clarification
    reply = synthesize("clarification", state, messages, [])
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)