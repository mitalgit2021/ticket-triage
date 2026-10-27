"""
AI Support Ticket Triage  (v3)
------------------------------
Reads support tickets, asks a local LLM (via Ollama) to classify each one,
validates the AI's answer, saves results to SQLite, and reports accuracy
against the correct labels in tickets.json.

v3 changes:
  - Schema-constrained output: Ollama is given the exact allowed values,
    so the model cannot invent categories like "account|technical".
  - Clearer output instructions in the prompt (no "a|b|c" format hint).
  - Retries include feedback about the invalid answer, so attempt 2
    is not an identical copy of attempt 1.
  - Raw model output is logged whenever validation fails.

Run:  python triage.py
Needs: Ollama running locally with the model pulled (ollama pull llama3.2)
"""

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Literal, Optional

import requests
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
TICKETS_FILE = "tickets.json"
DB_FILE = "triage.db"
MAX_ATTEMPTS = 2          # first try + one retry
REQUEST_TIMEOUT = 120     # seconds; local models can be slow on first load

# Double braces {{ }} are literal braces; {ticket} is filled in by .format().
# The example ticket is invented, NOT from tickets.json, to keep the test fair.
PROMPT_TEMPLATE = """You are a support triage assistant for a software company.
Classify the support ticket below.

CATEGORIES
- billing: charges, refunds, invoices, payment failures, plan changes, cancellations
- account: logging in, passwords, being locked out, account security, profile settings, adding or removing users
- technical: product bugs, crashes, errors, slowness, data displayed incorrectly
- other: pricing or discount questions, feature requests, legal or compliance questions, feedback

PRIORITY
- high: money charged wrongly, a system down, any security or data-privacy risk, a whole team blocked, or an urgent business deadline
- medium: something broken for one user, cancellations, legal or compliance requests
- low: how-to questions, feature requests, feedback, general questions

Judge priority by the real business impact, NOT the customer's tone.
"URGENT" or exclamation marks do not make a ticket high.
"No rush" does not make a serious problem low.

If a ticket fits two categories, pick the one the customer most needs resolved.

Reply ONLY with JSON.
"category" must be exactly ONE of: billing, technical, account, other.
"priority" must be exactly ONE of: low, medium, high.
Example: {{"category": "billing", "priority": "medium", "summary": "Customer asks why their plan price changed this month."}}

Ticket: {ticket}"""


# ---------------------------------------------------------------------------
# The shape we require from the AI. Anything else is rejected.
# ---------------------------------------------------------------------------
class Triage(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str


# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------
def load_tickets(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Talking to the AI
# ---------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    """Send a prompt to Ollama and return the raw text it replies with."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            # Give Ollama the exact schema, so the model can only produce
            # allowed values. (Older Ollama versions: update if this errors.)
            "format": Triage.model_json_schema(),
            "stream": False,                   # one complete reply
            "options": {"temperature": 0},     # same input -> same output
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def triage_ticket(text: str) -> Optional[Triage]:
    """Classify one ticket. Retry once with feedback; return None if it still fails."""
    base_prompt = PROMPT_TEMPLATE.format(ticket=text)
    prompt = base_prompt

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = call_llm(prompt)
            # Safety net: parse the JSON AND check fields/allowed values.
            return Triage.model_validate_json(raw)
        except ValidationError as e:
            print(f"    attempt {attempt}: AI output failed validation")
            print(f"      raw output: {raw}")
            for err in e.errors():
                print(f"      problem: {err['loc']} -> {err['msg']}")
            # Retry with feedback, so the next attempt isn't identical.
            prompt = (base_prompt
                      + f"\n\nYour previous answer was invalid: {raw}\n"
                      + "Fix it. Use exactly one allowed value for each field.")
        except requests.exceptions.Timeout:
            print(f"    attempt {attempt}: AI request timed out")

    return None  # caller marks this ticket for human review


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triage_results (
            ticket_id    INTEGER PRIMARY KEY,
            text         TEXT NOT NULL,
            category     TEXT,
            priority     TEXT,
            summary      TEXT,
            status       TEXT NOT NULL,   -- 'triaged' or 'needs_human_review'
            model        TEXT NOT NULL,
            latency_sec  REAL,
            processed_at TEXT NOT NULL
        )
    """)
    conn.commit()


def save_result(conn: sqlite3.Connection, ticket: dict,
                result: Optional[Triage], latency: float) -> None:
    # INSERT OR REPLACE + PRIMARY KEY = re-running never creates duplicates.
    conn.execute(
        """INSERT OR REPLACE INTO triage_results
           (ticket_id, text, category, priority, summary, status,
            model, latency_sec, processed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticket["id"],
            ticket["text"],
            result.category if result else None,
            result.priority if result else None,
            result.summary if result else None,
            "triaged" if result else "needs_human_review",
            MODEL,
            round(latency, 2),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------
def print_report(pairs: list[tuple[dict, Optional[Triage]]],
                 latencies: list[float]) -> None:
    total = len(pairs)
    triaged = [(t, r) for t, r in pairs if r is not None]
    review = total - len(triaged)

    cat_ok = sum(1 for t, r in triaged if r.category == t["true_category"])
    pri_ok = sum(1 for t, r in triaged if r.priority == t["true_priority"])
    both_ok = sum(1 for t, r in triaged
                  if r.category == t["true_category"]
                  and r.priority == t["true_priority"])

    def pct(n: int) -> str:
        return f"{n}/{total} ({100 * n / total:.0f}%)"

    print("\n" + "=" * 60)
    print("ACCURACY REPORT")
    print("=" * 60)
    print(f"Model:                 {MODEL}")
    print(f"Tickets:               {total}")
    print(f"Needs human review:    {review}")
    print(f"Category correct:      {pct(cat_ok)}")
    print(f"Priority correct:      {pct(pri_ok)}")
    print(f"Both correct:          {pct(both_ok)}")
    if latencies:
        print(f"Avg time per ticket:   {sum(latencies) / len(latencies):.1f}s")

    mistakes = [(t, r) for t, r in triaged
                if r.category != t["true_category"]
                or r.priority != t["true_priority"]]

    if mistakes:
        print("\nMISTAKES")
        print("-" * 60)
        for t, r in mistakes:
            print(f"#{t['id']}: {t['text']}")
            if r.category != t["true_category"]:
                print(f"    category: expected {t['true_category']}, got {r.category}")
            if r.priority != t["true_priority"]:
                print(f"    priority: expected {t['true_priority']}, got {r.priority}")

    if review:
        print("\nSENT TO HUMAN REVIEW")
        print("-" * 60)
        for t, r in pairs:
            if r is None:
                print(f"#{t['id']}: {t['text']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tickets = load_tickets(TICKETS_FILE)
    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    pairs: list[tuple[dict, Optional[Triage]]] = []
    latencies: list[float] = []

    print(f"Triaging {len(tickets)} tickets with {MODEL}...\n")
    try:
        for ticket in tickets:
            print(f"Ticket #{ticket['id']}")
            start = time.perf_counter()
            result = triage_ticket(ticket["text"])
            latency = time.perf_counter() - start

            save_result(conn, ticket, result, latency)
            pairs.append((ticket, result))
            latencies.append(latency)

            if result:
                print(f"    -> {result.category} / {result.priority}: {result.summary}")
            else:
                print("    -> needs human review")
    except requests.exceptions.ConnectionError:
        print("\nCould not reach Ollama at", OLLAMA_URL)
        print("Make sure Ollama is installed and running, then try again.")
        sys.exit(1)
    finally:
        conn.close()

    print_report(pairs, latencies)


if __name__ == "__main__":
    main()