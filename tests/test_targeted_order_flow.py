"""
tests/test_targeted_order_flow.py

Targeted regression tests for:
A. Multi-turn order follow-up context reuse (no duplicate lookup, no re-prompting for order ID)
B. Order ETA evidence selection (no generic domestic shipping policy contamination for ORD-1007)
C. Session isolation for unrelated queries following an order lookup
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

@pytest.fixture
def mock_retriever():
    ret = MagicMock()
    ret.retrieve.return_value = [
        {
            "filename": "05-domestic-shipping.md",
            "heading": "Delivery estimates after dispatch",
            "text": "Contiguous United States: 3-5 business days.",
            "score": 0.85,
            "active_official": True,
            "_matrix_idx": 0,
        },
        {
            "filename": "01-returns-policy-current.md",
            "heading": "Standard return window",
            "text": "Customers on the standard plan may request a return within 30 calendar days of delivery.",
            "score": 0.80,
            "active_official": True,
            "_matrix_idx": 1,
        }
    ]
    ret.matrix = None
    return ret

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock LLM generation
    def mock_generate(c, messages, config):
        resp = MagicMock()
        last_part = messages[-1].parts[0]
        text_in = getattr(last_part, "text", "") or ""
        
        if "ORD-1007" in text_in and not ("<ORDER_RESULT>" in text_in or "Active Order Reference" in text_in):
            # Model requests function call for ORD-1007
            part = MagicMock()
            fc = MagicMock()
            fc.name = "lookup_order"
            fc.args = {"order_id": "ORD-1007"}
            part.function_call = fc
            part.text = ""
            cand = MagicMock()
            cand.content.parts = [part]
            resp.candidates = [cand]
        elif "<ORDER_RESULT>" in text_in or "Active Order Reference" in text_in:
            # Model responds using order result
            part = MagicMock()
            part.function_call = None
            part.text = "For order ORD-1007, it is currently in transit with UPS, estimated to arrive on August 22, 2026."
            cand = MagicMock()
            cand.content.parts = [part]
            resp.candidates = [cand]
        elif "return window" in text_in.lower():
            part = MagicMock()
            part.function_call = None
            part.text = "The standard return window is within 30 calendar days of delivery per 01-returns-policy-current.md."
            cand = MagicMock()
            cand.content.parts = [part]
            resp.candidates = [cand]
        else:
            part = MagicMock()
            part.function_call = None
            part.text = "How can I help you?"
            cand = MagicMock()
            cand.content.parts = [part]
            resp.candidates = [cand]
        return resp

    return client, mock_generate

def test_order_eta_evidence_selection_no_domestic_policy_contamination(mock_retriever, mock_client):
    client, mock_gen = mock_client
    with patch("agent.agent.genai.Client"), \
         patch("agent.agent.load_chunks", return_value=[]), \
         patch("agent.agent.build_index", return_value=None), \
         patch("agent.agent.Retriever", return_value=mock_retriever), \
         patch("agent.agent._generate", side_effect=mock_gen):
        
        agent = Agent()
        session = get_or_create_session("test_session_eta_selection")
        
        res = agent.chat("Where is ORD-1007 and when should it arrive?", session)
        
        # Must not cite domestic shipping policy for specific Canadian order lookup
        assert not any("05-domestic-shipping.md" in src for src in res.sources), \
            f"05-domestic-shipping.md contaminated sources: {res.sources}"
        assert "August 22, 2026" in res.answer
        assert res.tool_called == "lookup_order"

def test_multi_turn_order_followup_context_reuse(mock_retriever, mock_client):
    client, mock_gen = mock_client
    with patch("agent.agent.genai.Client"), \
         patch("agent.agent.load_chunks", return_value=[]), \
         patch("agent.agent.build_index", return_value=None), \
         patch("agent.agent.Retriever", return_value=mock_retriever), \
         patch("agent.agent._generate", side_effect=mock_gen):
        
        agent = Agent()
        session = get_or_create_session("test_session_multi_turn")
        
        # Turn 1
        res1 = agent.chat("Where is ORD-1007 and when should it arrive?", session)
        assert res1.tool_called == "lookup_order"
        assert "August 22, 2026" in res1.answer

        # Turn 2: Follow-up query without explicit order ID
        with patch("agent.agent.lookup_order") as mock_lookup_call:
            res2 = agent.chat("When will it arrive?", session)
            
            # Must reuse existing session order context without making duplicate tool call
            mock_lookup_call.assert_not_called()
            assert "August 22, 2026" in res2.answer
            assert not res2.handoff

def test_unrelated_later_query_session_isolation(mock_retriever, mock_client):
    client, mock_gen = mock_client
    with patch("agent.agent.genai.Client"), \
         patch("agent.agent.load_chunks", return_value=[]), \
         patch("agent.agent.build_index", return_value=None), \
         patch("agent.agent.Retriever", return_value=mock_retriever), \
         patch("agent.agent._generate", side_effect=mock_gen):
        
        agent = Agent()
        session = get_or_create_session("test_session_isolation")

        # Turn 1: Order lookup
        res1 = agent.chat("Where is ORD-1007", session)
        assert not any("05-domestic-shipping.md" in src for src in res1.sources)

        # Turn 2: Unrelated policy question in same session
        res2 = agent.chat("What is the return window for a regular customer?", session)
        
        # Must answer return policy directly without asking for order ID or reusing ORD-1007
        assert "30 calendar days" in res2.answer
        assert not res2.handoff
