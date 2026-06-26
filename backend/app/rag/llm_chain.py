"""
SentinelSite — LLM Chain (RAG Answer Generation)
Strict grounding: answer ONLY from retrieved context.
Supports Anthropic (primary) + OpenAI (fallback).
"""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from app.config import settings
from app.db.models import DocumentType
from app.rag.intent_classifier import intent_classifier
from app.rag.retriever import RetrievedChunk, retriever

log = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
# Critical: strict grounding rules. No hallucination.

SYSTEM_PROMPT = """You are SentinelSite Voice Copilot — a safety assistant for construction workers.
You answer questions ONLY using the provided context from site documents.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. Always cite your source: "Per [filename], Page [N]: ..."
3. If the context does not contain the answer, respond EXACTLY:
   "I couldn't find that in the site documents. Please check with your supervisor."
4. Keep answers SHORT — workers are wearing glasses and cannot read long text.
   Maximum 3 sentences unless a safety procedure requires more steps.
5. For safety-critical information (fall protection, confined space, electrical hazards):
   Always end with "Verify with your supervisor before proceeding."
6. Never make assumptions or extrapolate beyond what the documents say.
7. Speak clearly and directly — no jargon unless the document uses it.
"""

# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    if not chunks:
        return "No relevant documents found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        page_str = f", Page {chunk.page_number}" if chunk.page_number else ""
        parts.append(
            f"[{i}] Source: {chunk.filename}{page_str}\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _build_user_message(query: str, context: str) -> str:
    return (
        f"CONTEXT FROM SITE DOCUMENTS:\n{context}\n\n"
        f"WORKER QUESTION: {query}"
    )


# ── LLM clients ───────────────────────────────────────────────────────────────

async def _call_anthropic(messages: list[dict], system: str) -> str:
    """Call Anthropic Claude."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
        system=system,
        messages=messages,
    )
    return response.content[0].text


async def _call_openai(messages: list[dict], system: str) -> str:
    """Call OpenAI GPT-4o-mini."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=settings.OPENAI_LLM_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=full_messages,
        temperature=0.1,   # low temp for factual answers
    )
    return response.choices[0].message.content


async def _call_llm(query: str, context: str) -> str:
    """
    Route to configured LLM provider.
    Falls back to OpenAI if Anthropic fails.
    """
    messages = [{"role": "user", "content": _build_user_message(query, context)}]

    if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        try:
            return await _call_anthropic(messages, SYSTEM_PROMPT)
        except Exception as e:
            log.warning(f"Anthropic call failed ({e}) — falling back to OpenAI")

    if settings.OPENAI_API_KEY:
        return await _call_openai(messages, SYSTEM_PROMPT)

    raise RuntimeError("No LLM provider configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")


# ── Answer result ─────────────────────────────────────────────────────────────

class RAGAnswer:
    def __init__(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        intent: DocumentType | None,
        intent_confidence: float,
        was_answered: bool,
        latency_ms: int,
        debug: dict[str, Any],
    ):
        self.answer = answer
        self.chunks = chunks
        self.intent = intent
        self.intent_confidence = intent_confidence
        self.was_answered = was_answered
        self.latency_ms = latency_ms
        self.debug = debug

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "was_answered": self.was_answered,
            "detected_intent": self.intent.value if self.intent else None,
            "intent_confidence": round(self.intent_confidence, 3),
            "latency_ms": self.latency_ms,
            "sources": [
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "chunk_text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "score": round(c.score, 4),
                    "page_number": c.page_number,
                }
                for c in self.chunks
            ],
            "debug": self.debug,
        }


# ── Main chain ────────────────────────────────────────────────────────────────

async def answer_query(
    site_id: str | UUID,
    query: str,
    force_doc_type: DocumentType | None = None,
) -> RAGAnswer:
    """
    End-to-end RAG pipeline:
    1. Classify intent → doc_type filter
    2. Hybrid search + rerank
    3. Build context
    4. LLM generation with strict grounding
    5. Return RAGAnswer

    Target: <3s end-to-end on LTE (NFR-R03)
    """
    t_start = time.perf_counter()

    # Step 1: Intent classification
    intent_result = intent_classifier.classify(query)
    doc_type = force_doc_type or (
        intent_result.doc_type
        if intent_result.doc_type != DocumentType.GENERAL
        else None   # GENERAL = no filter, search all docs
    )
    log.info(
        f"Query intent: {intent_result.label} ({intent_result.confidence:.2f}) "
        f"→ filter={doc_type}"
    )

    # Step 2: Retrieve
    chunks, retrieval_debug = await retriever.retrieve(
        site_id=site_id,
        query=query,
        doc_type_filter=doc_type,
        rerank=True,
    )

    # Step 3: Context
    context = _build_context(chunks)
    t_retrieve = time.perf_counter() - t_start

    # Step 4: Generate
    t_llm_start = time.perf_counter()
    try:
        answer = await _call_llm(query, context)
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        answer = "I'm unable to process your query right now. Please try again or ask your supervisor."

    t_llm = time.perf_counter() - t_llm_start
    t_total = time.perf_counter() - t_start

    # Determine if answered (not a "not found" response)
    was_answered = "couldn't find" not in answer.lower() and bool(chunks)

    debug = {
        **retrieval_debug,
        "intent_label": intent_result.label,
        "intent_confidence": round(intent_result.confidence, 3),
        "doc_type_filter": doc_type.value if doc_type else None,
        "retrieve_ms": round(t_retrieve * 1000),
        "llm_ms": round(t_llm * 1000),
        "total_ms": round(t_total * 1000),
        "llm_provider": settings.LLM_PROVIDER,
    }

    log.info(
        f"RAG answered: was_answered={was_answered}, "
        f"total={debug['total_ms']}ms "
        f"(retrieve={debug['retrieve_ms']}ms, llm={debug['llm_ms']}ms)"
    )

    return RAGAnswer(
        answer=answer,
        chunks=chunks,
        intent=intent_result.doc_type,
        intent_confidence=intent_result.confidence,
        was_answered=was_answered,
        latency_ms=round(t_total * 1000),
        debug=debug,
    )
