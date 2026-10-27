# AI Support Ticket Triage

Automatically classifies customer support tickets by **category** and **priority** using a local LLM, validates every answer, routes failures to human review, and measures accuracy against labeled data.

## The business problem

Support teams sort hundreds of tickets a day by hand. Urgent issues ("our production system is down") can wait in the same queue as "how do I change my profile picture?". This pipeline triages each ticket automatically so urgent work reaches the right people first.

## How it works

```
tickets.json -> LLM classifies -> schema validation -> retry with feedback -> human review if still invalid
                                                 |
                       accuracy report  <-  SQLite (triage.db)
```

- **Local LLM via Ollama** (free, runs on a laptop, no data leaves the machine)
- **Schema-constrained output**: the model is given the exact allowed values, and Pydantic validates the reply as a second check
- **Retry with feedback**: if an answer is invalid, the model is told what was wrong; if it still fails, the ticket goes to `needs_human_review` instead of being guessed
- **Idempotent storage**: re-running updates rows in SQLite instead of duplicating them
- **Evaluation**: category, priority and combined accuracy, a list of every mistake, and average latency per ticket
- **Security guardrail** (v4): a second, narrow AI check asks only whether a ticket describes a security or data-privacy risk, and can raise priority to high

## Results

### Prompt iterations (llama3.2, main set of 20 tickets)

| Version | Category | Priority | Both | Human review |
|---|---|---|---|---|
| v1: basic prompt | 60% | 75% | 40% | 0 |
| v2: category and priority definitions | 65% | 50% | 45% | 5 |
| v3: schema-constrained output | 85% | 70% | 60% | 0 |

### Model comparison (v3 prompt, no guardrail)

| Model | Main set (20) | Held-out set (10) | Combined (30) |
|---|---|---|---|
| llama3.2 (3B) | 60% both | 80% both | 67% |
| qwen2.5:7b | 75% both | 80% both | 77% |

qwen2.5:7b reached 95% category accuracy on the main set and fixed llama3.2's consistent habit of labeling invoices, cancellations and plan changes as "account".

## Key findings

1. **The model copied our format hint.** v2 sent 25% of tickets to human review, every one of them high priority. Logging the raw output showed the model returning values like `"account|technical"`, copied from the `"billing|technical|account|other"` format line in the prompt. Fixed at the source with schema-constrained output.
2. **Retries were useless at first.** With temperature 0 and an identical prompt, the retry produced the exact same invalid answer. Retries now include feedback about what was wrong.
3. **Errors have different costs.** Most priority mistakes over-rated tickets, which only means faster handling. But both models under-rated a calmly worded data-exposure ticket (another company's records visible). That kind of miss could become a legal incident, which motivated the security guardrail.
4. **Test sets differ in difficulty.** llama3.2 scored higher on the held-out set than on the main set, because the main set was deliberately loaded with ambiguous tickets. A score only means something next to the data it was measured on.

## Limitations and next steps

- The security guardrail is implemented but **not yet evaluated**. It was designed after seeing specific failures, so it must be measured on new security-related tickets before anyone relies on it. It also doubles the number of AI calls per ticket.
- 30 labeled tickets is a small sample: each ticket is 3-5% of a score. A real deployment needs a few hundred of the customer's own tickets, labeled by their support team.
- Some labels are genuinely debatable (for example, "paid for premium but features don't work": billing or technical?). These need agreed rules with the support team, not more prompt tuning.
- Latency figures from a single run include model loading time and should be re-measured warm.

## Run it yourself

Requirements: Python 3.10+, [Ollama](https://ollama.com)

```bash
ollama pull qwen2.5:7b

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python triage.py                       # main test set (20 tickets)
python triage.py tickets_holdout.json  # held-out test set (10 tickets)
```

Settings at the top of `triage.py`: `MODEL` to switch models, `USE_GUARDRAIL` to turn the security check on or off.

## Files

| File | Purpose |
|---|---|
| `triage.py` | The pipeline: classification, validation, guardrail, storage, report |
| `tickets.json` | 20 labeled tickets, including deliberately ambiguous ones |
| `tickets_holdout.json` | 10 labeled tickets never used for prompt tuning |
