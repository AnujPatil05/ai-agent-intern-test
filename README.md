# Aster & Row — Reliable RAG Support Agent

## 1. Overview

Built a context-aware, reliable RAG support agent for Aster & Row (an ecommerce company selling bags, drinkware, and travel accessories) using Python, Gemini 3.6 Flash, and a deterministic safety layer. The system handles customer inquiries, policy questions, multi-turn order tracking, and human support handoffs. It explicitly addresses four major LLM failure modes: false policy conflicts across differently scoped rules, loss of multi-turn session order context, hallucinated/unallowed source citations, and evidence contamination from generic shipping policies during order tracking.

---

## 2. Architecture

```text
User Input ──► Conversation State ──► Intent & Retrieval ──► LLM Response Generation
                      │                      │                        │
                      ▼                      ▼                        ▼
               Session History        Order Tool / RAG      Deterministic Validation
               & Order Context        Vector Search        (Safety & Citation Filter)
                                                                      │
                                                                      ▼
                                                                Final Response
```

* **Conversation State (`agent/conversation.py`)**: Manages multi-turn history and caches active session order lookups (`last_order_id` & `last_order_result`).
* **Retrieval & Ingestion (`agent/ingestion.py`, `agent/retriever.py`)**: Indexes policy documents with Gemini embeddings (`gemini-embedding-2`), enforces metadata precedence (`active_official` over `superseded`), and detects genuine policy conflicts via a cosine-similarity gated matrix.
* **Order Tool (`agent/order_tool.py`)**: Authoritative lookup engine for `orders.json` with PII sanitization and stale field suppression.
* **Agent Core (`agent/agent.py`)**: Orchestrates tool calls, context building, multi-turn order follow-ups, and generic policy filtering.
* **Deterministic Safety & Validation (`agent/safety.py`)**: Performs pre-LLM evidence validation, post-LLM PII/injection checks, and regex sanitization of unallowed source citations.

---

## 3. Tech Stack

* **LLM / Model**: Google Gemini 3.6 Flash (`google-genai` SDK).
* **Embedding Approach**: `gemini-embedding-2` (768-dimensional normalized dense vectors).
* **Framework & Libraries**: Python 3.11+, `numpy` (vector dot-product matrix operations), `pytest`, `python-dotenv`.
* **Storage & Index**: Local JSON & NumPy disk cache (`.cache/embeddings.json`).
* **Why these fit**: Provides cold startup (<1s), zero external database dependencies, deterministic vector similarity gating, and high reliability within the 6–8 hour timebox.

---

## 4. Setup

```bash

pip install -r requirements.txt


cp .env.example .env



python cli.py
```

---

## 5. Evaluation

### Commands

```bash

python evaluate.py


python evaluate_robustness.py
```

### Baseline Results (Development / Pre-Triage)

| Suite | Baseline Score | Category Breakdown (Development Baseline) |
|---|:---:|---|
| **Standard Suite (`evaluate.py`)** | **13 / 22 (59%)** | Abstention: 1/1 (100%)<br>Conversation: 1/2 (50%)<br>Groundedness: 2/2 (100%)<br>Multi-Source Grounding: 0/2 (0%)<br>Privacy: 1/2 (50%)<br>Prompt-Security: 0/1 (0%)<br>Retrieval: 1/3 (33%)<br>Source-Conflict: 1/1 (100%)<br>Tool-Reliability: 4/5 (80%)<br>Tool-Use: 2/3 (67%) |

Most of the failed cases are related to semantic errors , spelling errors.

---

## 6. Bug Diary

### Bug 1: False Conflict Between Differently Scoped Policies
* **Reproduction**: Querying standard return window ("What is the return window for a regular customer?") or Canadian order ETA ("Where is ORD-1007?") caused agent to declare a false contradiction and trigger unnecessary handoff.
* **Root Cause**: `CONFLICT_TOPIC_THRESHOLD` cosine gate was unread in `retriever.py`, and `_num_units_conflict` collapsed distinct unit types (`calendar days` vs `business days` vs `reporting days`), treating unrelated policies as contradictory.
* **Fix**: Implemented matrix cosine-similarity gating (`matrix[idx1] @ matrix[idx2] >= 0.55`) in `detect_conflicts()` and preserved unit sub-types (`calendar_day` vs `business_day`).
* **Regression Test**: `tests/test_conflict_detection.py` (`test_no_conflict_with_cosine_gate`).

### Bug 2: Lost Multi-Turn Order Context
* **Reproduction**: Turn 1: "Where is ORD-1007 and when should it arrive?" (Order lookup succeeded). Turn 2: "When will it arrive?" -> Agent prompted "Please share your order ID...".
* **Root Cause**: `Session` dataclass lacked persistence for `last_order_result`, and agent routing failed to reuse active session order context on follow-up queries.
* **Fix**: Added `last_order_result` to `Session`, implemented `_is_order_followup()` context detector, and reused cached order results without redundant API lookups.
* **Regression Test**: `tests/test_targeted_order_flow.py` (`test_multi_turn_order_followup_context_reuse`).

### Bug 3: Fabricated / Unallowed Source Citation
* **Reproduction**: Requesting private data ("What is the customer's email and risk score?") returned a refusal that cited `07-customer-privacy-and-data-protection.md § Data protection` (a document not present in the 14-file KB).
* **Root Cause**: LLM generated a generic refusal citation, and `validate_response()` lacked source integrity checks comparing cited `.md` filenames against `retrieved` chunks.
* **Fix**: Created `sanitize_unallowed_citations(text, allowed_chunks)` in `agent/safety.py` to strip unretrieved `.md` filename citations from raw LLM responses.
* **Regression Test**: `tests/test_targeted_privacy.py` (`test_privacy_request_refuses_sensitive_data_and_strips_fabricated_sources`).

---

## 7. Safety & Reliability

* **Metadata-Aware Source Precedence**: Filters out `superseded` policies and migration notes; only `active_official` chunks serve as authoritative evidence.
* **Conflict Detection**: Vector cosine-similarity gate combined with negation-scope matching surfaces genuine contradictions (e.g. Breeze Tumbler dishwasher care) while suppressing false positives.
* **Order-Data Sanitization**: Strips sensitive customer PII (email, address, risk score, internal notes) from order lookup output before feeding context to LLM.
* **Stale-Field Suppression**: Hides carrier, tracking number, and ETA on cancelled or returned orders to prevent misleading claims.
* **Prompt-Injection Handling**: Treats retrieved KB passages and order details as untrusted context; resists instruction overrides (e.g., Doc 14 prompt injection attempts).
* **Abstention & Handoff**: Triggers human handoff for genuine policy conflicts, missing KB evidence, exception order statuses, or unsupported user action requests.
* **Citation Validation**: Post-LLM regex layer strips hallucinated or unallowed `.md` filenames not present in retrieved evidence.

---

## 8. Known Limitations

* **In-Memory Session State**: Active session context and cached order results are stored in memory and reset upon CLI process restart.
* **Phrase-Based Abstention Detection**: Relies on `_abstain_phrases` string matching rather than structured LLM boolean signals.
* **Rule-Based Evidence Filtering**: Filters generic shipping policy filenames (`05-domestic-shipping.md`, `06-international-shipping.md`) for active order status queries to prevent evidence contamination.

---

## 9. AI Coding Tools Used

* **Tools Used**: Antigravity IDE / AI Agent assistant used for pair programming, refactoring retrieval logic, and writing targeted pytest suites.
* **Incorrect AI Suggestion & Manual Fix**: The AI initially suggested stripping all numerical day comparison logic in `_num_units_conflict` to fix false conflict detections. Manual testing revealed this broke genuine numeric conflict detection for identical policy scopes; the solution was refined to preserve unit sub-types (`calendar_day` vs `business_day`) and enforce the vector cosine gate.

---

## 10. Demo

![Aster & Row Agent Demo](demo.gif)

The recorded demo demonstrates:
1. Knowledge-base query with accurate file & section citations.
2. Direct order lookup using `lookup_order` tool.
3. Multi-turn contextual follow-up without re-asking for order ID.
4. PII refusal and automatic human handoff recommendation.
5. Evaluation suite execution.

---

## 11. Repository Structure

```text
.
├── agent/
│   ├── agent.py               # Main agent orchestration & LLM interaction
│   ├── conversation.py        # Session management & turn tracking
│   ├── ingestion.py           # Ingestion & embedding caching
│   ├── order_tool.py          # Authoritative order lookup & sanitization
│   ├── retriever.py           # Vector retrieval & conflict detection
│   └── safety.py              # Pre/post-LLM safety & citation validation
├── data/
│   ├── orders-data-dictionary.md
│   └── orders.json
├── evaluation/
│   └── visible-cases.json     # Standard visible evaluation cases
├── knowledge-base/            # 14 markdown policy documents
├── tests/                     # Pytest regression suite
│   ├── test_conflict_detection.py
│   ├── test_embed_retry.py
│   ├── test_targeted_order_flow.py
│   └── test_targeted_privacy.py
├── cli.py                     # Interactive CLI interface
├── evaluate.py                # Main evaluation suite
├── evaluate_robustness.py     # Robustness evaluation suite
├── demo.gif                   # Recorded demonstration
├── requirements.txt
└── README.md
```
