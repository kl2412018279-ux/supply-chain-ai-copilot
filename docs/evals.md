# Copilot answer-quality evals

The copilot answers money questions: quotes in RM, demurrage exposure, margins, carrier
on-time rates. The main risk with an AI assistant here is a confident answer with a wrong
number in it. The eval suite measures that risk, along with tool choice and guardrails, on
a fixed set of golden questions. It runs offline in CI on every push, and against a real
LLM when you have an API key.

```bash
python -m src.evals.run --agent rule --fail-under 0.9    # offline, what CI runs
python -m src.evals.run --category quote -v              # one category, case by case
```

It writes `reports/evals.json` (every case, with answer, tools called and failures) and
`reports/evals.md` (a summary), and prints a table per category. The **AI quality** page
(`views/evals.py`) shows the latest report and has a **Run evaluation** button.

## The golden set

`data/evals/copilot_golden.jsonl` holds one question per line:

| Field | Meaning |
|---|---|
| `id`, `category` | e.g. `quote-01`, `quote` |
| `question` | what a user would type |
| `expected_tools` | tools the answer needs (order does not matter) |
| `must_include` | text that must appear, case-insensitive; prefix `re:` for a regex |
| `expect_blocked` | `true` for attacks the guardrails must refuse |
| `notes` | why the case exists |

There are 30 normal questions across quoting, mode comparison, route and carrier
recommendation, delay risk, free time and demurrage, margin leakage, forecasting, volume
history, spend projection, haulier scorecards and the operations manual. There are also 10
adversarial ones: instruction overrides, requests for the system prompt or API keys, fake
role tags, a persona jailbreak, off-topic questions, SQL injection and two attempts to pull
customers' personal data.

When the agent fails a case, fix the agent, not the case. Only rewrite a question if it
is genuinely ambiguous, and keep it a question a real user would ask.

## What each metric means

**Tool selection accuracy.** The share of normal questions where the agent called every
tool the question needs. Calling an extra tool doesn't fail a case, but it lowers
**precision** (the share of called tools that were expected). **Recall** is the share of
expected tools that were called. If the agent picks the wrong tool, the answer can't be
right, however fluent it reads.

**Groundedness (numeric faithfulness).** The share of answers in which *every* RM amount,
percentage and number with a unit (km, TEU, L, h, days, containers...) can be traced back
to something the agent was given. That means the tool outputs and arguments it received,
or the user's own question. Before comparing, the check normalises formatting and rounds
to the precision the answer shows:

- `RM1,234.50`, `RM 1234.5` and `RM1.2k` are read as numbers. `RM1,234.50` matches a
  tool value between 1234.495 and 1234.505. `RM1,235` matches anything that rounds to it.
- `91.2%` matches a tool value of `91.2` or a stored fraction of `0.912`.
- A number inside tool text counts too, for example "3 days" in a manual snippet or
  "RM 300,000" in a policy.
- Simple arithmetic counts as support: the sum or difference of two fields in the same
  record (revenue minus cost), and column totals over a list of records, overall or
  grouped by a text field (quote line items summed per cost category).
- The sign is ignored, so "2 days over" matches `days_left = -2`.

Any number left over is reported as **unsupported**, case by case. This is how the suite
catches hallucinated numbers. Bare numbers without a unit (ranks, years, form numbers,
"fit 77/100") are not checked. **Numeric claims supported** gives the same check at claim
level, across all answers.

How sensitive it is: as a check, every number in the offline agent's answers was scaled by
±3-10% and the result checked again. About 98% of the changed RM amounts were flagged,
along with about 85% of changed quantities and about 80% of changed percentages. Small
whole numbers, like "2 d" or "15%", sometimes match an unrelated tool value by
coincidence. So a pass here means "no evidence of invented numbers", not proof that every
figure is right.

**must_include pass rate.** The share of answered normal questions that contain all the
required text, such as the destination name, "RM", "TEU" or the K8 form from the manual.
It is a cheap check that the answer is about the right thing.

**Guardrail block rate.** The share of adversarial questions that were refused before
reaching the agent. **False-block rate** is the share of normal questions that were
refused by mistake. Both matter: a guardrail that blocks everything is safe but useless.

**Latency p50 / p95.** The median and 95th-percentile time per question through the full
pipeline (guardrails, agent and tools, redaction), with the semantic cache off.

**Pass rate.** The share of cases that pass every check that applies to them.

**LLM judge (optional).** When an LLM provider is configured, the same model rates each
answered question at temperature 0 on two 1-5 scales. **Relevance** asks whether it
answers what was asked. **Helpfulness** asks whether a dispatcher could act on it. The
prompt is in `src/evals/judge.py`. It asks for JSON only and tells the judge not to reward
length. Offline the judge is skipped. Treat the scores as a trend signal, not a gate:
judges are biased towards fluent text and can't check arithmetic. The groundedness check
covers that.

## CI gate

`--fail-under 0.9` makes the command exit with status 1 when overall **tool accuracy** or
**groundedness** is below 90%. CI runs it after the tests, in the Python 3.12 job, with the
offline rule-based agent. Exit status 2 means the run couldn't start, for example because
`--agent llm` was requested with no LLM configured, or the category doesn't exist.

## Running against Groq

You need `langchain-groq`, which is in `requirements.txt` but not in `requirements-ci.txt`.

```bash
LLM_PROVIDER=groq GROQ_API_KEY=gsk_... python -m src.evals.run --agent llm
```

- `--agent llm` fails fast if no LLM is configured. `--agent auto` (the default) uses the
  LLM agent when one is configured and the offline agent otherwise.
- The judge turns on automatically when a provider is configured. Use `--judge off` to
  save tokens, or `--judge on` to require it.
- On free-tier rate limits, add `--delay 2` to pause between questions.
- LLM answers vary from run to run, so compare runs over time rather than chasing a single
  point. Set `LLM_MODEL` to compare models on the same golden set.
- The eval doesn't fall back to the rule-based agent when the LLM errors. A provider
  failure shows up as failing cases, not as a quietly good score.
