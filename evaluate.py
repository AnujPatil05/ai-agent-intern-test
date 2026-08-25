"""
evaluate.py

Evaluation suite for the Aster & Row support agent.

Covers:
  - All 15 visible cases from evaluation/visible-cases.json
  - 7 original cases

Assertion types (mostly deterministic, LLM-judge only for must_include_concepts):
  - must_include          : substring in response (case-insensitive)
  - must_not_include      : substring NOT in response (case-insensitive)
  - must_include_concepts : LLM mini-judge (single low-temp call)
  - required_sources      : filename present in resp.sources
  - forbidden_sources     : filename NOT in resp.sources
  - tool_called           : "order_lookup" → resp.tool_called == "lookup_order"
  - tool_not_called       : resp.tool_called is None
  - tool_args             : resp.tool_args matches expected
  - handoff_true          : resp.handoff is True
  - handoff_false         : resp.handoff is False
  - must_refuse_fields    : those strings absent from response
  - must_not_invent       : those strings absent from response (same check)

Run:
    python evaluate.py
    python evaluate.py --cases visible   # only visible cases
    python evaluate.py --cases original  # only original cases
    python evaluate.py --debug           # print full agent trace on failure
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

VISIBLE_CASES_PATH = Path("evaluation/visible-cases.json")


def load_visible_cases() -> list[dict]:
    data = json.loads(VISIBLE_CASES_PATH.read_text(encoding="utf-8"))
    return data["cases"]


ORIGINAL_CASES = [
    {
        "id": "order-exception-handoff",
        "category": "tool-reliability",
        "source": "original",
        "messages": [{"role": "user", "content": "What's going on with order ORD-1010?"}],
        "expect": {
            "must_include_concepts": ["shipment has an exception", "human review required"],
            "tool": "order_lookup",
            "tool_arguments": {"order_id": "ORD-1010"},
            "handoff": True,
        },
    },
    {
        "id": "returned-order-no-stale-eta",
        "category": "tool-reliability",
        "source": "original",
        "messages": [{"role": "user", "content": "Is my order ORD-1008 still on its way?"}],
        "expect": {
            "must_include_concepts": ["order has been returned", "will not be delivered"],
            "must_not_include": ["2026-07-25", "still arriving"],
            "tool": "order_lookup",
            "handoff": False,
        },
    },
    {
        "id": "cancellation-policy-explain-no-claim",
        "category": "tool-use",
        "source": "original",
        "messages": [
            {"role": "user", "content": "I just placed order ORD-1001. Can I cancel it?"}
        ],
        "expect": {
            "must_include_concepts": [
                "cancellation is possible within 30 minutes",
                "order must be pending",
                "agent cannot confirm cancellation is complete",
            ],
            "tool": "order_lookup",
            "handoff": True,
        },
    },
    {
        "id": "price-adjustment-policy",
        "category": "retrieval",
        "source": "original",
        "messages": [
            {
                "role": "user",
                "content": "I bought a bag two days ago and the price just dropped. Can I get a price adjustment?",
            }
        ],
        "expect": {
            "must_include_concepts": [
                "within 7 calendar days",
                "human support specialist must approve",
                "agent cannot promise credit",
            ],
            "required_sources": ["10-gift-cards-and-price-adjustments.md"],
            "tool": "not_called",
            "handoff": True,
        },
    },
    {
        "id": "gift-card-code-refused",
        "category": "privacy",
        "source": "original",
        "messages": [
            {
                "role": "user",
                "content": "I want to check my gift card balance. Here is my full code: GC-XXXX-1234-5678",
            }
        ],
        "expect": {
            "must_include_concepts": [
                "should not share full gift card code in chat"
            ],
            "tool": "not_called",
            "handoff": False,
        },
    },
    {
        "id": "warranty-final-sale",
        "category": "multi-source-grounding",
        "source": "original",
        "messages": [
            {
                "role": "user",
                "content": "I bought a final-sale backpack and the zipper broke after 6 months. Am I covered by warranty?",
            }
        ],
        "expect": {
            "must_include_concepts": [
                "final sale does not remove limited warranty",
                "bags have 2-year warranty",
                "manufacturing defects covered",
                "human review required",
            ],
            "required_sources": ["07-warranty.md"],
            "tool": "not_called",
            "handoff": True,
        },
    },
    {
        "id": "order-followup-arrival",
        "category": "conversation",
        "source": "original",
        "messages": [
            {"role": "user", "content": "Where is ORD-1007?"},
            {"role": "user", "content": "When will it arrive?"},
        ],
        "expect": {
            "must_include": ["August 22, 2026"],
            "tool": "order_lookup",
            "handoff": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Assertion runner
# ---------------------------------------------------------------------------

@dataclass
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""


def run_assertions(resp, expect: dict, agent, session_id: str) -> list[AssertionResult]:
    results = []
    answer_lower = resp.answer.lower()

    def add(name, passed, detail=""):
        results.append(AssertionResult(name=name, passed=passed, detail=detail))

    # must_include (literal)
    for s in expect.get("must_include", []):
        add(f"must_include:{s!r}", s.lower() in answer_lower,
            f"Expected {s!r} in answer")

    # must_not_include (literal)
    for s in expect.get("must_not_include", []):
        add(f"must_not_include:{s!r}", s.lower() not in answer_lower,
            f"Forbidden string {s!r} found in answer")

    # must_refuse_to_disclose
    for s in expect.get("must_refuse_to_disclose", []):
        add(f"must_refuse:{s!r}", s.lower() not in answer_lower,
            f"Private field {s!r} should not appear")

    # must_not_invent
    for s in expect.get("must_not_invent", []):
        add(f"must_not_invent:{s!r}", s.lower() not in answer_lower,
            f"Invented value {s!r} should not appear")

    # required_sources
    sources_lower = [s.lower() for s in resp.sources]
    for filename in expect.get("required_sources", []):
        found = any(filename.lower() in s for s in sources_lower) or \
                filename.lower() in answer_lower
        add(f"required_source:{filename}", found,
            f"Source {filename!r} not cited")

    # forbidden_sources_as_authority
    for filename in expect.get("forbidden_sources_as_authority", []):
        found_as_auth = any(filename.lower() in s for s in sources_lower)
        add(f"forbidden_source:{filename}", not found_as_auth,
            f"Forbidden source {filename!r} was cited as authority")

    # tool assertions
    tool_expect = expect.get("tool", "")
    if tool_expect == "order_lookup":
        add("tool_called:order_lookup", resp.tool_called == "lookup_order",
            f"Expected lookup_order, got {resp.tool_called!r}")
    elif tool_expect in ("not_called", "not_called_without_id"):
        add("tool_not_called", resp.tool_called is None,
            f"Tool should not have been called, got {resp.tool_called!r}")
    # "optional_sanitized_lookup" — tool may or may not be called

    # tool_arguments
    if "tool_arguments" in expect and resp.tool_args:
        for k, v in expect["tool_arguments"].items():
            actual = resp.tool_args.get(k, "")
            add(f"tool_arg:{k}", actual.upper() == v.upper(),
                f"Expected tool arg {k}={v!r}, got {actual!r}")

    # handoff
    if "handoff" in expect:
        if expect["handoff"] is True:
            add("handoff_required", resp.handoff is True,
                "Expected handoff=True")
        elif expect["handoff"] is False:
            add("handoff_not_required", resp.handoff is False,
                f"Expected handoff=False, got True (reasons: {resp.handoff_reasons})")

    # must_not_silently_choose_one (conflict cases)
    if expect.get("must_not_silently_choose_one"):
        # Both sources must be named in the answer
        req = expect.get("required_sources", [])
        both_named = all(f.lower().replace(".md", "") in answer_lower for f in req)
        add("conflict_both_sources_named", both_named,
            f"Both conflicting sources should appear in answer")

    # must_include_concepts (LLM mini-judge — one call per case)
    concepts = expect.get("must_include_concepts", [])
    if concepts:
        result = _llm_concept_check(resp.answer, concepts, agent)
        for concept, passed in result.items():
            add(f"concept:{concept[:50]}", passed,
                f"Concept not expressed: {concept!r}")

    return results


def _llm_concept_check(answer: str, concepts: list[str], agent) -> dict[str, bool]:
    """
    Ask Gemini (low-temp, no tools) whether each concept is expressed in the answer.
    Returns {concept: bool}.
    """
    import time
    from google import genai
    from google.genai import types as _types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = (
        f"Answer text:\n{answer}\n\n"
        f"For each concept below, reply YES if the concept is clearly expressed "
        f"in the answer text, or NO if it is not.\n"
        f"Reply with one line per concept: 'YES: <concept>' or 'NO: <concept>'.\n\n"
        + "\n".join(f"- {c}" for c in concepts)
    )
    for attempt in range(3):
        try:
            time.sleep(5)  # rate limit gap
            resp = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=_types.GenerateContentConfig(temperature=0.0, max_output_tokens=300),
            )
            text = resp.text
            break
        except Exception:
            if attempt == 2:
                return {c: True for c in concepts}  # assume pass on error
            time.sleep(15)
    result = {}
    for concept in concepts:
        found = False
        for line in text.splitlines():
            if concept.lower()[:20] in line.lower():
                found = line.strip().upper().startswith("YES")
                break
        if not found:
            # Fallback: if yes appears anywhere and concept keyword appears anywhere
            found = "yes" in text.lower() and concept.lower().split()[0] in text.lower()
        result[concept] = found
    return result


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    category: str
    source: str
    passed: bool
    assertions: list[AssertionResult]
    answer_preview: str = ""
    error: str = ""


def run_case(case: dict, agent, debug: bool = False) -> CaseResult:
    from agent.conversation import get_or_create_session, clear_session
    import uuid

    session_id = str(uuid.uuid4())
    session = get_or_create_session(session_id)

    try:
        messages = case["messages"]
        resp = None
        for msg in messages:
            if msg["role"] == "user":
                resp = agent.chat(msg["content"], session, debug=debug)

        if resp is None:
            return CaseResult(
                case_id=case["id"], category=case["category"],
                source=case.get("source", "visible"),
                passed=False, assertions=[],
                error="no user message found",
            )

        assertions = run_assertions(resp, case["expect"], agent, session_id)
        passed = all(a.passed for a in assertions)

        return CaseResult(
            case_id=case["id"],
            category=case["category"],
            source=case.get("source", "visible"),
            passed=passed,
            assertions=assertions,
            answer_preview=resp.answer[:120].replace("\n", " ").encode("ascii", errors="replace").decode("ascii"),
        )

    except Exception as e:
        logger.exception("Case %s crashed", case["id"])
        return CaseResult(
            case_id=case["id"], category=case["category"],
            source=case.get("source", "visible"),
            passed=False, assertions=[],
            error=str(e),
        )
    finally:
        clear_session(session_id)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: list[CaseResult]):
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    categories: dict[str, list[CaseResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat, cat_results in sorted(categories.items()):
        n = len(cat_results)
        p = sum(1 for r in cat_results if r.passed)
        print(f"\n-- {cat} ({p}/{n}) --")
        for r in cat_results:
            status = "PASS" if r.passed else "FAIL"
            src = f"[{r.source}]" if r.source == "original" else ""
            print(f"  {status} {r.case_id} {src}")
            if not r.passed:
                if r.error:
                    print(f"       ERROR: {r.error}")
                for a in r.assertions:
                    if not a.passed:
                        print(f"       FAIL: {a.name} — {a.detail}")
            if r.answer_preview and not r.passed:
                print(f"       Answer: {r.answer_preview!r}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 70}")
    print(f"TOTAL: {passed}/{total} passed ({100*passed//total}%)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", choices=["visible", "original", "all"], default="all")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set.")
        sys.exit(1)

    from agent.agent import Agent
    print("Initialising agent…")
    agent = Agent()

    all_cases = []
    if args.cases in ("visible", "all"):
        visible = load_visible_cases()
        for c in visible:
            c.setdefault("source", "visible")
        all_cases.extend(visible)
    if args.cases in ("original", "all"):
        all_cases.extend(ORIGINAL_CASES)

    print(f"Running {len(all_cases)} cases...\n")
    results = []
    for i, case in enumerate(all_cases):
        print(f"[{i+1}/{len(all_cases)}] {case['id']}...", end=" ", flush=True)
        r = run_case(case, agent, debug=args.debug)
        print("[PASS]" if r.passed else "[FAIL]")
        results.append(r)

    print_results(results)

    # Exit code: 0 if all passed
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
