"""
agent/ingestion.py

Parse knowledge-base markdown files, extract YAML front-matter,
chunk by section heading, embed with Gemini gemini-embedding-2,
and cache embeddings to disk so subsequent runs are instant.

Document eligibility rules (enforced here — not delegated to LLM):
  - status: superseded   → ineligible as evidence
  - status: draft        → ineligible as evidence
  - policy_authority: none → ineligible as evidence
  - customer_answering: false → ineligible as evidence
  - audience: internal   → ineligible as *customer-facing* evidence

Only active + official + customer chunks reach the LLM synthesis step.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import yaml
from google.genai import errors as _genai_errors

logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent.parent / "knowledge-base"
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_FILE = CACHE_DIR / "embeddings.json"

_EMBEDDING_MODEL = "models/gemini-embedding-2"
_SUPERSEDED_OR_DRAFT = {"superseded", "draft"}


def _is_per_day_quota(e) -> bool:
    """
    Return True when a ClientError/ServerError is a daily quota exhaustion.
    Mirrors the helper in retriever.py; both walk e.details (parsed JSON dict).
    """
    try:
        details_list = e.details.get("error", {}).get("details", [])
        return any("PerDay" in str(d.get("quotaId", "")) for d in details_list)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Return (metadata_dict, body_text) from a Markdown file."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm = content[3:end].strip()
    body = content[end + 4:].strip()
    try:
        meta = yaml.safe_load(fm) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, body


# ---------------------------------------------------------------------------
# Eligibility classification
# ---------------------------------------------------------------------------

def classify(meta: dict) -> dict:
    """
    Add computed eligibility flags to a metadata dict (mutates in-place).
    Keys added:
      _eligible         — may be used as evidence at all
      _active_official  — active + official + customer audience
      _superseded       — explicitly superseded
    """
    status = str(meta.get("status", "")).lower()
    authority = str(meta.get("policy_authority", "")).lower()
    audience = str(meta.get("audience", "customer")).lower()
    customer_answering = meta.get("customer_answering", True)

    superseded = (status == "superseded")
    eligible = (
        status not in _SUPERSEDED_OR_DRAFT
        and authority != "none"
        and customer_answering is not False
        and audience != "internal"
    )
    active_official = (
        status == "active"
        and authority == "official"
        and audience == "customer"
    )

    meta["_eligible"] = eligible
    meta["_active_official"] = active_official
    meta["_superseded"] = superseded
    return meta


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(body: str, meta: dict, filename: str) -> list[dict]:
    """Split document body on ## headings; each chunk carries full metadata."""
    sections = re.split(r"\n(?=## )", body)
    chunks = []
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().splitlines()
        heading_line = lines[0].strip()
        heading = heading_line.lstrip("#").strip() if heading_line.startswith("#") else "Overview"
        text = section.strip()
        if len(text) < 30:
            continue
        chunks.append({
            "filename": filename,
            "heading": heading,
            "text": text,
            "metadata": meta,
            "eligible": meta["_eligible"],
            "active_official": meta["_active_official"],
        })
    return chunks


# ---------------------------------------------------------------------------
# Load all chunks (no embeddings)
# ---------------------------------------------------------------------------

def load_chunks() -> list[dict]:
    chunks = []
    for md_file in sorted(KB_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(content)
        meta["_filename"] = md_file.name
        classify(meta)
        for chunk in chunk_document(body, meta, md_file.name):
            chunks.append(chunk)
    logger.info("loaded %d chunks from %s", len(chunks), KB_DIR)
    return chunks


# ---------------------------------------------------------------------------
# Embedding + caching (google-genai SDK)
# ---------------------------------------------------------------------------

def _corpus_hash(chunks: list[dict]) -> str:
    combined = "".join(c["text"] for c in chunks)
    return hashlib.md5(combined.encode()).hexdigest()


def embed_texts(texts: list[str], client) -> np.ndarray:
    """
    Embed a list of texts using Gemini gemini-embedding-2.
    `client` is a google.genai.Client instance.

    Each text is embedded with up to 3 attempts and exponential backoff
    (1 s, 4 s) for transient 429/503 errors.  PerDay quota exhaustion
    raises RuntimeError immediately without retrying.
    """
    _MAX_ATTEMPTS = 3
    embeddings = []
    for text in texts:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                result = client.models.embed_content(
                    model=_EMBEDDING_MODEL,
                    contents=text[:8000],
                )
                embeddings.append(result.embeddings[0].values)
                break  # success — move to next text
            except (_genai_errors.ClientError, _genai_errors.ServerError) as e:
                if _is_per_day_quota(e):
                    raise RuntimeError(
                        "embed_texts: daily quota exhausted — "
                        "cannot retry a PerDay limit"
                    ) from e
                if e.code in (429, 503) and attempt < _MAX_ATTEMPTS - 1:
                    delay = 4 ** attempt  # 1 s, 4 s
                    logger.warning(
                        "embed_texts: API error %s (attempt %d/%d), "
                        "retrying in %ds",
                        e.code, attempt + 1, _MAX_ATTEMPTS, delay,
                    )
                    time.sleep(delay)
                else:
                    raise
    return np.array(embeddings, dtype=np.float32)


def build_index(chunks: list[dict], client) -> np.ndarray:
    """
    Return embedding matrix for all chunks.
    Uses a JSON cache so repeated runs are free.
    `client` is a google.genai.Client instance.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    corpus_hash = _corpus_hash(chunks)

    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if cached.get("hash") == corpus_hash:
            logger.info("embedding cache hit (%d vectors)", len(cached["embeddings"]))
            return np.array(cached["embeddings"], dtype=np.float32)

    logger.info("building embeddings for %d chunks…", len(chunks))
    texts = [f"{c['heading']}\n{c['text']}" for c in chunks]
    matrix = embed_texts(texts, client)

    CACHE_FILE.write_text(
        json.dumps({"hash": corpus_hash, "embeddings": matrix.tolist()}),
        encoding="utf-8",
    )
    logger.info("embeddings cached to %s", CACHE_FILE)
    return matrix
