"""
cli.py — Interactive CLI for the Aster & Row support agent.

Usage:
    python cli.py              # normal mode
    python cli.py --debug      # show full trace after each response

Type 'new' to start a fresh session, 'quit' to exit.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

def _setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _print_response(resp, debug: bool):
    print("\n" + "-" * 60)
    print(resp.answer)

    if resp.sources:
        print("\n[src] Sources:")
        for s in resp.sources:
            print(f"   • {s}")

    if resp.handoff:
        print("\n[!]  Human handoff recommended.")
        for r in resp.handoff_reasons:
            print(f"   Reason: {r}")

    if resp.tool_called:
        print(f"\n[tool] Tool called: {resp.tool_called}({resp.tool_args})")

    if debug and resp.debug:
        import json
        print("\n-- DEBUG TRACE --")
        print(json.dumps(resp.debug, indent=2, default=str))

    print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent")
    parser.add_argument("--debug", action="store_true", help="Show full trace")
    args = parser.parse_args()

    _setup_logging(args.debug)

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    from agent.agent import Agent
    from agent.conversation import get_or_create_session, clear_session

    print("Initialising agent (first run builds embeddings — takes ~30s)…")
    agent = Agent()
    session = get_or_create_session()
    print(f"Session {session.id[:8]}… started. Type 'new' for a new session, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "new":
            clear_session(session.id)
            session = get_or_create_session()
            print(f"New session {session.id[:8]}… started.\n")
            continue

        resp = agent.chat(user_input, session, debug=args.debug)
        _print_response(resp, args.debug)


if __name__ == "__main__":
    main()
