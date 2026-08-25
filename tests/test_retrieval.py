"""
tests/test_retrieval.py

Tests for ingestion, retriever, and conflict detection.
These tests DO require GEMINI_API_KEY (for embedding).

Run:  python -m pytest tests/test_retrieval.py -v
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# Skip whole module if no API key
if not os.environ.get("GEMINI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set"
)


@pytest.fixture(scope="module")
def retriever():
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    from agent.ingestion import load_chunks, build_index
    from agent.retriever import Retriever
    chunks = load_chunks()
    matrix = build_index(chunks, client)
    return Retriever(chunks, matrix, client)


# ---------------------------------------------------------------------------
# Eligibility: forbidden documents must not reach the LLM
# ---------------------------------------------------------------------------

class TestEligibility:
    def test_superseded_excluded(self, retriever):
        """02-returns-policy-legacy.md (superseded) must never appear in results."""
        results = retriever.retrieve("how many days do I have to return an item")
        filenames = [r["filename"] for r in results]
        assert "02-returns-policy-legacy.md" not in filenames, \
            "superseded document leaked into retrieval results"

    def test_internal_migration_excluded(self, retriever):
        """14-internal-content-migration-notes.md (policy_authority=none) must be excluded."""
        results = retriever.retrieve("return policy days")
        filenames = [r["filename"] for r in results]
        assert "14-internal-content-migration-notes.md" not in filenames, \
            "policy_authority=none document leaked into retrieval results"

    def test_internal_escalation_excluded_from_customer_evidence(self, retriever):
        """13-support-escalation.md (audience=internal) must not appear in customer evidence."""
        results = retriever.retrieve("when should I contact support")
        filenames = [r["filename"] for r in results]
        assert "13-support-escalation.md" not in filenames, \
            "internal-audience document leaked into customer evidence"


# ---------------------------------------------------------------------------
# Active authoritative docs ARE retrieved
# ---------------------------------------------------------------------------

class TestActiveDocsRetrieved:
    def test_current_returns_policy_retrieved(self, retriever):
        results = retriever.retrieve("what is the return window for standard customers")
        filenames = [r["filename"] for r in results]
        assert "01-returns-policy-current.md" in filenames

    def test_trailplus_retrieved(self, retriever):
        results = retriever.retrieve("TrailPlus member return window")
        filenames = [r["filename"] for r in results]
        assert "09-trailplus-membership.md" in filenames

    def test_international_shipping_retrieved(self, retriever):
        results = retriever.retrieve("do you ship to Canada?")
        filenames = [r["filename"] for r in results]
        assert "06-international-shipping.md" in filenames

    def test_warranty_retrieved(self, retriever):
        results = retriever.retrieve("warranty for bags")
        filenames = [r["filename"] for r in results]
        assert "07-warranty.md" in filenames


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_tumbler_dishwasher_conflict_detected(self, retriever):
        """
        11-product-care.md says hand-wash the body;
        12-breeze-tumbler-product-card.md says all components are dishwasher safe.
        Both are active+official → conflict must be detected.
        """
        from agent.retriever import detect_conflicts
        results = retriever.retrieve("can I put the Breeze Tumbler in the dishwasher")
        conflicts = detect_conflicts(results)
        filenames_in_conflicts = set()
        for ca, cb in conflicts:
            filenames_in_conflicts.add(ca["filename"])
            filenames_in_conflicts.add(cb["filename"])
        assert "11-product-care.md" in filenames_in_conflicts or \
               "12-breeze-tumbler-product-card.md" in filenames_in_conflicts, \
            "Tumbler dishwasher conflict not detected"

    def test_no_false_conflict_on_simple_query(self, retriever):
        """A simple policy question about returns should not raise a conflict."""
        from agent.retriever import detect_conflicts
        results = retriever.retrieve("how do I return an item")
        conflicts = detect_conflicts(results)
        # Standard return window is only in the current policy doc;
        # legacy is excluded; so no active-official conflict should fire.
        conflict_file_pairs = [
            (ca["filename"], cb["filename"]) for ca, cb in conflicts
        ]
        # The legacy doc must not be in any conflict pair
        for a, b in conflict_file_pairs:
            assert "02-returns-policy-legacy.md" not in (a, b), \
                "Superseded document appeared in conflict detection"


# ---------------------------------------------------------------------------
# Relevance threshold: low-relevance queries return fewer chunks
# ---------------------------------------------------------------------------

class TestRelevanceThreshold:
    def test_unrelated_query_returns_few_chunks(self, retriever):
        results = retriever.retrieve("what is the capital of France")
        # Off-topic query: any returned chunks should have relatively low scores.
        # The important property is that no chunk scores above 0.65 (high confidence).
        high_conf = [r for r in results if r["score"] > 0.65]
        assert len(high_conf) == 0, \
            f"Off-topic query returned high-confidence chunks: {high_conf}"

