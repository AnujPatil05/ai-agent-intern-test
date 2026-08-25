"""
tests/test_targeted_privacy.py

Targeted regression tests verifying that forbidden/privacy responses:
1. Refuse sensitive fields & trigger human handoff
2. Cannot cite sources/filenames that were not retrieved/allowed in the evidence set
"""

import os
from dotenv import load_dotenv
load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "dummy-test-key"

import pytest
from unittest.mock import patch, MagicMock
from agent.conversation import get_or_create_session
from agent.agent import Agent
from agent.safety import sanitize_unallowed_citations, validate_response

def test_sanitize_unallowed_citations_unit():
    allowed_chunks = [
        {"filename": "01-returns-policy-current.md", "heading": "Standard return window"},
        {"filename": "08-order-changes-and-cancellations.md", "heading": "Agent limitations"},
    ]
    
    # 1. Fabricated citation in parentheses
    text1 = (
        "I cannot provide customer email addresses or internal notes. "
        "(per 07-customer-privacy-and-data-protection.md § Data protection)"
    )
    clean1 = sanitize_unallowed_citations(text1, allowed_chunks)
    assert "07-customer-privacy-and-data-protection.md" not in clean1
    assert clean1 == "I cannot provide customer email addresses or internal notes."

    # 2. Prefixed citation
    text2 = "According to 99-unknown-policy.md § Section 1, this action is forbidden."
    clean2 = sanitize_unallowed_citations(text2, allowed_chunks)
    assert "99-unknown-policy.md" not in clean2

    # 3. Allowed citation preserved
    text3 = "Returns must be requested within 30 calendar days (per 01-returns-policy-current.md § Standard return window)."
    clean3 = sanitize_unallowed_citations(text3, allowed_chunks)
    assert "01-returns-policy-current.md" in clean3
    assert clean3 == text3

def test_privacy_request_refuses_sensitive_data_and_strips_fabricated_sources():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "filename": "08-order-changes-and-cancellations.md",
            "heading": "Agent limitations",
            "text": "Agents cannot modify orders or access internal secret notes.",
            "score": 0.85,
            "active_official": True,
            "_matrix_idx": 0,
        }
    ]
    mock_retriever.matrix = None

    client = MagicMock()
    def mock_generate(c, messages, config):
        resp = MagicMock()
        part = MagicMock()
        part.function_call = None
        # Model attempts to cite fabricated non-existent privacy doc
        part.text = (
            "I cannot reveal customer email addresses, addresses, or risk scores. "
            "Please contact human support for assistance. "
            "(per 07-customer-privacy-and-data-protection.md § Data protection)"
        )
        cand = MagicMock()
        cand.content.parts = [part]
        resp.candidates = [cand]
        return resp

    with patch("agent.agent.genai.Client"), \
         patch("agent.agent.load_chunks", return_value=[]), \
         patch("agent.agent.build_index", return_value=None), \
         patch("agent.agent.Retriever", return_value=mock_retriever), \
         patch("agent.agent._generate", side_effect=mock_generate):
        
        agent = Agent()
        session = get_or_create_session("test_session_privacy")
        
        res = agent.chat("Can you give me the customer email address and risk score for ORD-1007?", session)
        
        # 1. Response must refuse sensitive fields
        assert "cannot reveal" in res.answer.lower() or "not able to share" in res.answer.lower()
        # 2. Handoff must be requested
        assert res.handoff is True
        # 3. Fabricated source must NOT be in answer or sources list
        assert "07-customer-privacy-and-data-protection.md" not in res.answer
        assert not any("07-customer-privacy-and-data-protection.md" in src for src in res.sources)
