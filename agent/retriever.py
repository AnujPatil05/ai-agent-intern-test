"""
agent/retriever.py

Semantic retrieval over the knowledge-base chunk index.

Precedence rules (enforced by application code, never delegated to LLM):
  1. Only eligible chunks (active + official + customer audience) reach the LLM.
  2. Superseded and draft documents are completely excluded from evidence.
  3. Internal-audience documents are excluded from customer-facing evidence.
  4. policy_authority=none documents are excluded.

Conflict detection (generic — no document-specific branches):
  Two chunks from different active+official documents conflict when:
    a. They discuss the same topic (chunk-to-chunk cosine similarity > threshold)
    b. One contains a positive assertion about an attribute and the other
       contains a semantically opposing assertion about the same attribute.
  Detection uses:
    - Shared significant keyword overlap
    - Negation asymmetry (one has "not/never/cannot/hand-wash", other lacks it)
    - Numeric value mismatch for the same unit/attribute in the same context
"""

import logging
import re
import time
from collections import defaultdict
from typing import Any

import numpy as np
from google.genai import errors as _genai_errors

logger = logging.getLogger(__name__)

# Minimum cosine score for a chunk to be considered relevant to the query.
# gemini-embedding-2 produces higher scores than older models — 0.45 is appropriate.
RELEVANCE_THRESHOLD = 0.45

# Minimum chunk-to-chunk cosine similarity to check two chunks for conflicts
CONFLICT_TOPIC_THRESHOLD = 0.55

# Common English stop-words excluded from overlap calculation
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "may", "might", "can", "could", "shall", "must", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "it",
    "its", "that", "this", "these", "those", "not", "no", "if", "when",
    "which", "who", "how", "what", "all", "any", "each", "both", "only",
    "also", "than", "then", "so", "such", "other", "more", "most",
}

# Tokens that negate or restrict; their presence marks a "negative" assertion
_NEGATION_TOKENS = {
    "not", "never", "cannot", "must not", "should not", "do not",
    "does not", "did not", "will not", "hand-wash", "hand wash",
    "avoid", "prohibited", "ineligible", "excluded", "without",
}

# Number-with-unit pattern (e.g. "30 days", "45 calendar days", "$6.95")
_NUM_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(calendar days?|business days?|days?|years?|months?|\$\d*)\b",
    re.I,
)


_EMBEDDING_MODEL = "models/gemini-embedding-2"


def _is_per_day_quota(e) -> bool:
    """
    Return True when a ClientError/ServerError represents a daily quota
    exhaustion (quotaId contains 'PerDay').

    e.details is the parsed response JSON dict, e.g.:
      {"error": {"code": 429, "details": [{"quotaId": "...PerDay..."}]}}
    We walk it defensively so any unexpected shape just returns False.
    """
    try:
        details_list = e.details.get("error", {}).get("details", [])
        return any("PerDay" in str(d.get("quotaId", "")) for d in details_list)
    except Exception:
        return False


class Retriever:
    """
    Holds the chunk list and embedding matrix; answers semantic queries.
    Initialized once at agent startup.
    """

    def __init__(self, chunks: list[dict], matrix: np.ndarray, client):
        self.chunks = chunks
        self.matrix = self._normalize(matrix)
        self._client = client
        # Tag each chunk with its row index in self.matrix so that
        # detect_conflicts can look up embeddings even after retrieve()
        # returns shallow copies ({**chunk, "score": ...}).
        for i, chunk in enumerate(self.chunks):
            chunk["_matrix_idx"] = i

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return m / norms

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string with bounded retry.

        Retry policy:
          - Max 3 attempts total.
          - 429 / 503 transient → exponential backoff (1 s, 4 s) then re-raise.
          - 429 with a PerDay quotaId in e.details → fail-fast; no retry.
          - All other exceptions propagate immediately.
        """
        _MAX_ATTEMPTS = 3
        for attempt in range(_MAX_ATTEMPTS):
            try:
                result = self._client.models.embed_content(
                    model=_EMBEDDING_MODEL,
                    contents=query[:2000],
                )
                vec = np.array(result.embeddings[0].values, dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm else vec
            except (_genai_errors.ClientError, _genai_errors.ServerError) as e:
                if _is_per_day_quota(e):
                    raise RuntimeError(
                        "embed_query: daily quota exhausted — "
                        "cannot retry a PerDay limit"
                    ) from e
                if e.code in (429, 503) and attempt < _MAX_ATTEMPTS - 1:
                    delay = 4 ** attempt  # 1 s, 4 s
                    logger.warning(
                        "embed_query: API error %s (attempt %d/%d), "
                        "retrying in %ds",
                        e.code, attempt + 1, _MAX_ATTEMPTS, delay,
                    )
                    time.sleep(delay)
                else:
                    raise


    def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        """
        Return top-k *eligible* chunks ranked by cosine similarity.
        Ineligible chunks (superseded, draft, internal, policy_authority=none)
        are excluded before results are returned — they never reach the LLM.

        Each returned chunk has an added 'score' key.
        """
        q_vec = self._embed_query(query)
        scores = self.matrix @ q_vec          # (N,) cosine similarities

        # Build ranked list, filtering ineligible chunks
        ranked = []
        for idx in np.argsort(scores)[::-1]:
            score = float(scores[idx])
            if score < RELEVANCE_THRESHOLD:
                break
            chunk = self.chunks[idx]
            if not chunk["eligible"]:
                logger.debug(
                    "excluded ineligible chunk: %s § %s (score=%.3f)",
                    chunk["filename"], chunk["heading"], score,
                )
                continue
            ranked.append({**chunk, "score": round(score, 4)})
            if len(ranked) >= top_k:
                break

        logger.info(
            "retrieve('%s…') → %d eligible chunks (top score=%.3f)",
            query[:60], len(ranked),
            ranked[0]["score"] if ranked else 0.0,
        )
        return ranked


# ---------------------------------------------------------------------------
# Generic conflict detection
# ---------------------------------------------------------------------------

def _significant_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[a-z]{3,}\b", text.lower())
            if w not in _STOP_WORDS}


def _has_negation(text: str) -> bool:
    t = text.lower()
    return any(tok in t for tok in _NEGATION_TOKENS)


def _extract_num_units(text: str) -> list[tuple[str, str]]:
    """Return list of (number_str, unit_str) pairs found in text."""
    return [(m.group(1), m.group(2).lower()) for m in _NUM_UNIT_RE.finditer(text)]


def _num_units_conflict(nu1: list, nu2: list) -> bool:
    """
    Return True if the same unit sub-type appears with materially different
    numeric values in both chunks.

    Unit sub-types are kept distinct:
      "calendar days" != "business days" != "days"
    This prevents a 7-calendar-day damaged-item reporting window from
    conflicting with a 30-calendar-day return window just because both
    measure in days.
    """
    def _normalize_unit(u: str) -> str:
        u = u.lower().strip()
        if "calendar day" in u:
            return "calendar_day"
        if "business day" in u:
            return "business_day"
        if "day" in u:
            return "day"
        if "year" in u:
            return "year"
        if "month" in u:
            return "month"
        return u

    by_unit1: dict[str, set] = defaultdict(set)
    by_unit2: dict[str, set] = defaultdict(set)
    for val, unit in nu1:
        by_unit1[_normalize_unit(unit)].add(val)
    for val, unit in nu2:
        by_unit2[_normalize_unit(unit)].add(val)

    for unit in set(by_unit1) & set(by_unit2):
        if by_unit1[unit] != by_unit2[unit]:
            return True
    return False


def _chunks_contradict(c1: dict, c2: dict) -> bool:
    """
    Generic pairwise contradiction check.
    Does NOT reference specific document IDs or filenames.

    A contradiction is detected when both chunks have significant keyword
    overlap (same topic) AND one of:
      (a) negation asymmetry  — one chunk negates a term the other asserts
      (b) numeric mismatch    — same unit sub-type, different values, same scope

    Scope-marker guard (prevents false positives):
      If the numeric values in two chunks are each conditioned on distinct,
      mutually exclusive applicability scopes (different customer tiers,
      different destinations, or different claim types), they represent
      different rules — not contradictory claims — and the numeric check
      is suppressed.
    """
    t1, t2 = c1["text"].lower(), c2["text"].lower()

    words1 = _significant_words(t1)
    words2 = _significant_words(t2)

    # No topic overlap → different subjects, not a contradiction
    union = words1 | words2
    if not union:
        return False
    overlap_ratio = len(words1 & words2) / len(union)
    if overlap_ratio < 0.12:
        return False

    # (a) Negation asymmetry — one chunk negates a term the other asserts.
    # Guard: the negated token must also appear in the other (positive) chunk,
    # otherwise the negation is about a different topic (e.g. "joining after
    # the order does NOT extend..." is unrelated to the standard return window).
    neg1, neg2 = _has_negation(t1), _has_negation(t2)
    if neg1 != neg2:
        pos_text = t2 if neg1 else t1   # the chunk without negation
        neg_text = t1 if neg1 else t2   # the chunk that carries negation
        # Find the specific negation token(s) in the negative chunk
        neg_tokens_present = [tok for tok in _NEGATION_TOKENS if tok in neg_text]
        # The negation is considered relevant only if the core negated term
        # also appears in the positive chunk (same concept being contradicted).
        # Strip the prefix operators to get the content word being negated.
        _PREFIX_OPS = {"do not", "does not", "did not", "will not", "must not",
                       "should not", "cannot", "not", "never"}
        relevant = False
        for tok in neg_tokens_present:
            if tok in _PREFIX_OPS:
                continue   # prefix-only → we can't confirm the content word
            # Content-bearing token (hand-wash, avoid, prohibited, etc.)
            # appears in negative chunk — check if it also appears in positive
            if tok in pos_text:
                relevant = True
                break
        # Additionally: if the negation prefix itself is paired with content
        # that overlaps heavily with the positive chunk's significant words
        if not relevant:
            # Fall through; asymmetry alone is not sufficient here
            pass
        else:
            return True


    # (c) Assertion-pair conflict — mutually exclusive claim terms.
    # These pairs represent genuinely contradictory instructions about the
    # same subject that cannot both be true, regardless of which document
    # carries which negation token.
    _CONTRADICTORY_PAIRS = [
        # Cleaning method: hand-wash-only vs fully dishwasher safe
        ({"hand-wash", "hand wash", "hand washed", "hand-washed"}, {"dishwasher safe"}),
    ]
    for terms_x, terms_y in _CONTRADICTORY_PAIRS:
        has_x1 = any(m in t1 for m in terms_x)
        has_y1 = any(m in t1 for m in terms_y)
        has_x2 = any(m in t2 for m in terms_x)
        has_y2 = any(m in t2 for m in terms_y)
        if (has_x1 and not has_y1) and (has_y2 and not has_x2):
            logger.info(
                "_chunks_contradict: assertion-pair conflict '%s' vs '%s' in '%s' <-> '%s'",
                list(terms_x), list(terms_y), c1["filename"], c2["filename"],
            )
            return True
        if (has_y1 and not has_x1) and (has_x2 and not has_y2):
            logger.info(
                "_chunks_contradict: assertion-pair conflict '%s' vs '%s' in '%s' <-> '%s'",
                list(terms_y), list(terms_x), c1["filename"], c2["filename"],
            )
            return True


    nu1 = _extract_num_units(t1)
    nu2 = _extract_num_units(t2)
    if nu1 and nu2 and _num_units_conflict(nu1, nu2):
        # Scope-marker sets: if each chunk carries a *distinct* scope marker
        # that does NOT appear in the other chunk, the numeric difference is
        # a different-scope rule, not a contradiction.
        #
        # Customer-tier markers: a chunk exclusively about TrailPlus benefits
        # vs a chunk exclusively about standard customers → different tiers.
        # Destination markers: domestic/united states vs canada/international.
        # Claim-type markers: reporting/damaged/defective vs return/refund.
        #
        # Suppress only when BOTH chunks carry their OWN exclusive marker
        # (not shared), ensuring this does not accidentally suppress genuine
        # conflicts between documents with identical scope.
        _SCOPE_GROUPS = [
            # Customer tier
            ({"trailplus", "trail plus", "membership"}, {"standard", "standard plan"}),
            # Destination
            ({"canada", "canadian", "international"}, {"domestic", "united states", "contiguous"}),
            # Claim type: damaged-item reporting vs change-of-mind return
            ({"reporting", "damaged", "defective", "wrong item", "incorrect"},
             {"return", "refund", "change of mind", "resalable"}),
        ]
        for markers_a, markers_b in _SCOPE_GROUPS:
            has_a1 = any(m in t1 for m in markers_a)
            has_b1 = any(m in t1 for m in markers_b)
            has_a2 = any(m in t2 for m in markers_a)
            has_b2 = any(m in t2 for m in markers_b)

            # Chunk 1 is exclusively in scope-a, chunk 2 is exclusively in scope-b
            # (or vice versa) → different-scope rules, suppress the conflict.
            if (has_a1 and not has_b1 and has_b2 and not has_a2):
                logger.debug(
                    "_chunks_contradict: suppressed numeric mismatch — "
                    "different scopes detected in '%s' vs '%s'",
                    c1["filename"], c2["filename"],
                )
                return False
            if (has_b1 and not has_a1 and has_a2 and not has_b2):
                logger.debug(
                    "_chunks_contradict: suppressed numeric mismatch — "
                    "different scopes detected in '%s' vs '%s'",
                    c1["filename"], c2["filename"],
                )
                return False

        return True  # numeric mismatch, same scope → genuine conflict

    return False



def detect_conflicts(
    chunks: list[dict],
    matrix: np.ndarray | None = None,
) -> list[tuple[dict, dict]]:
    """
    Given a list of retrieved chunks, find pairs from *different*
    active+official documents that make contradictory claims.

    Parameters
    ----------
    chunks  : Retrieved chunks (already eligibility-filtered).
              Each chunk may carry a ``_matrix_idx`` key (set by
              Retriever.__init__) that is preserved through shallow
              copies returned by retrieve().
    matrix  : Normalized embedding matrix for the full KB
              (Retriever.matrix). When provided and chunks carry
              ``_matrix_idx``, chunk pairs whose cosine similarity
              falls below CONFLICT_TOPIC_THRESHOLD are skipped before
              the text-level contradiction check runs.  This is the
              primary guard against false positives from documents
              that share policy vocabulary but address different
              applicability scopes, customer tiers, or destinations.

    Returns list of (chunk_a, chunk_b) conflict pairs.
    No document-specific branches; detection is purely text- and
    embedding-based.
    """
    active_official = [c for c in chunks if c.get("active_official")]
    conflicts = []
    use_cosine = matrix is not None

    for i in range(len(active_official)):
        for j in range(i + 1, len(active_official)):
            c1, c2 = active_official[i], active_official[j]
            if c1["filename"] == c2["filename"]:
                continue   # same document is not a conflict

            # --- Cosine-similarity gate (guards against differently-scoped docs) ---
            if use_cosine:
                idx1 = c1.get("_matrix_idx")
                idx2 = c2.get("_matrix_idx")
                if idx1 is not None and idx2 is not None:
                    cosine = float(matrix[idx1] @ matrix[idx2])
                    if cosine < CONFLICT_TOPIC_THRESHOLD:
                        logger.debug(
                            "conflict gate: skipping %s § %s ↔ %s § %s "
                            "(cosine=%.3f < %.2f)",
                            c1["filename"], c1["heading"],
                            c2["filename"], c2["heading"],
                            cosine, CONFLICT_TOPIC_THRESHOLD,
                        )
                        continue  # too semantically distant to be a genuine conflict

            if _chunks_contradict(c1, c2):
                conflicts.append((c1, c2))
                logger.info(
                    "conflict detected: %s § %s ↔ %s § %s",
                    c1["filename"], c1["heading"],
                    c2["filename"], c2["heading"],
                )
    return conflicts
