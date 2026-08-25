"""
tests/test_embed_retry.py

Unit tests for the bounded retry wrapper added to:
  - agent/retriever.py :: Retriever._embed_query
  - agent/ingestion.py :: embed_texts
  - The shared _is_per_day_quota helper (tested via both modules)

No GEMINI_API_KEY required — all SDK calls are mocked.

Run:
    python -m pytest tests/test_embed_retry.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch, call
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers: build fake SDK error objects that match the real APIError shape
# ---------------------------------------------------------------------------

def _make_api_error(cls, code: int, quota_id: str | None = None):
    """
    Build a fake ClientError / ServerError whose attributes match what
    the real google.genai.errors.APIError sets:
        e.code   -> int HTTP status
        e.details -> dict matching the Gemini error response body
    """
    if quota_id:
        details = {"error": {"code": code, "details": [{"quotaId": quota_id}]}}
    else:
        details = {"error": {"code": code, "details": []}}

    err = cls.__new__(cls)
    err.code = code
    err.details = details
    err.message = f"Fake error {code}"
    err.status = "RESOURCE_EXHAUSTED" if code == 429 else "UNAVAILABLE"
    return err


def _client_error(code: int, quota_id: str | None = None):
    from google.genai.errors import ClientError
    return _make_api_error(ClientError, code, quota_id)


def _server_error(code: int):
    from google.genai.errors import ServerError
    return _make_api_error(ServerError, code)


def _good_embed_result(dim: int = 4):
    """Fake embed_content return value with a unit vector."""
    result = MagicMock()
    result.embeddings = [MagicMock()]
    result.embeddings[0].values = [1.0] + [0.0] * (dim - 1)
    return result


# ---------------------------------------------------------------------------
# Tests for _is_per_day_quota (retriever.py)
# ---------------------------------------------------------------------------

class TestIsPerDayQuota:
    def test_returns_true_for_per_day_quota_id(self):
        from agent.retriever import _is_per_day_quota
        e = _client_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        assert _is_per_day_quota(e) is True

    def test_returns_false_for_per_minute_quota_id(self):
        from agent.retriever import _is_per_day_quota
        e = _client_error(429, "GenerateRequestsPerMinutePerProjectPerModel")
        assert _is_per_day_quota(e) is False

    def test_returns_false_for_503(self):
        from agent.retriever import _is_per_day_quota
        e = _server_error(503)
        assert _is_per_day_quota(e) is False

    def test_returns_false_for_broken_details(self):
        from agent.retriever import _is_per_day_quota
        e = MagicMock()
        e.details = None          # not a dict — should not raise
        assert _is_per_day_quota(e) is False

    def test_ingestion_helper_matches(self):
        """Both modules define the same logic; spot-check ingestion's copy."""
        from agent.ingestion import _is_per_day_quota as ing_helper
        from agent.retriever import _is_per_day_quota as ret_helper
        e = _client_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        assert ing_helper(e) == ret_helper(e) == True


# ---------------------------------------------------------------------------
# Tests for Retriever._embed_query
# ---------------------------------------------------------------------------

class TestEmbedQueryRetry:

    def _make_retriever(self, mock_client):
        from agent.retriever import Retriever
        chunks = []
        matrix = np.zeros((0, 4), dtype=np.float32)
        r = Retriever(chunks, matrix, mock_client)
        return r

    def test_success_on_first_attempt(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _good_embed_result()
        r = self._make_retriever(mock_client)

        vec = r._embed_query("hello")

        assert mock_client.models.embed_content.call_count == 1
        assert vec.shape == (4,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5  # unit vector

    def test_retries_once_on_transient_429(self):
        from google.genai.errors import ClientError
        transient = _client_error(429)   # no PerDay quotaId

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = [
            transient,
            _good_embed_result(),
        ]
        r = self._make_retriever(mock_client)

        with patch("agent.retriever.time.sleep") as mock_sleep:
            vec = r._embed_query("hello")

        assert mock_client.models.embed_content.call_count == 2
        mock_sleep.assert_called_once_with(1)   # 4**0 == 1 s
        assert vec.shape == (4,)

    def test_retries_twice_on_two_transient_503s(self):
        from google.genai.errors import ServerError
        transient = _server_error(503)

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = [
            transient,
            transient,
            _good_embed_result(),
        ]
        r = self._make_retriever(mock_client)

        with patch("agent.retriever.time.sleep") as mock_sleep:
            vec = r._embed_query("hello")

        assert mock_client.models.embed_content.call_count == 3
        assert mock_sleep.call_args_list == [call(1), call(4)]  # 4**0, 4**1

    def test_raises_after_three_transient_failures(self):
        from google.genai.errors import ClientError
        transient = _client_error(429)

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = [transient] * 3
        r = self._make_retriever(mock_client)

        with patch("agent.retriever.time.sleep"):
            with pytest.raises(ClientError):
                r._embed_query("hello")

        assert mock_client.models.embed_content.call_count == 3

    def test_fail_fast_on_per_day_quota(self):
        """PerDay quota must NOT be retried — RuntimeError raised immediately."""
        per_day = _client_error(
            429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = per_day
        r = self._make_retriever(mock_client)

        with patch("agent.retriever.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="daily quota exhausted"):
                r._embed_query("hello")

        # Only one attempt, no sleep
        assert mock_client.models.embed_content.call_count == 1
        mock_sleep.assert_not_called()

    def test_non_429_503_error_propagates_immediately(self):
        """A 400 Bad Request should not be retried."""
        from google.genai.errors import ClientError
        bad_request = _client_error(400)

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = bad_request
        r = self._make_retriever(mock_client)

        with patch("agent.retriever.time.sleep") as mock_sleep:
            with pytest.raises(ClientError):
                r._embed_query("hello")

        assert mock_client.models.embed_content.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for ingestion.embed_texts (same logic, different function)
# ---------------------------------------------------------------------------

class TestEmbedTextsRetry:

    def test_success_no_retries(self):
        from agent.ingestion import embed_texts
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = _good_embed_result(4)

        result = embed_texts(["text a", "text b"], mock_client)

        assert mock_client.models.embed_content.call_count == 2
        assert result.shape == (2, 4)

    def test_fail_fast_on_per_day_quota(self):
        from agent.ingestion import embed_texts
        per_day = _client_error(
            429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = per_day

        with patch("agent.ingestion.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="daily quota exhausted"):
                embed_texts(["text a"], mock_client)

        assert mock_client.models.embed_content.call_count == 1
        mock_sleep.assert_not_called()

    def test_retries_per_chunk_independently(self):
        """
        First chunk succeeds immediately.
        Second chunk fails once (transient 429) then succeeds.
        Total embed_content calls: 1 + 2 = 3.
        """
        from agent.ingestion import embed_texts
        transient = _client_error(429)

        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = [
            _good_embed_result(4),   # chunk 0: success
            transient,               # chunk 1: first attempt fails
            _good_embed_result(4),   # chunk 1: second attempt succeeds
        ]

        with patch("agent.ingestion.time.sleep") as mock_sleep:
            result = embed_texts(["text a", "text b"], mock_client)

        assert mock_client.models.embed_content.call_count == 3
        mock_sleep.assert_called_once_with(1)
        assert result.shape == (2, 4)
