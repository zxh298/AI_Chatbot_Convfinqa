# ConvFinQA Report

## Summary

v1 = full-record baseline
v2 = full-record baseline + record-local evidence selection
v3 = evidence selection + no-gold verification retry
v4 = evidence selection + no-gold verification retry + train-example reasoning retrieval
v5 = evidence selection + no-gold verification retry + structured calculation-plan execution
v5a = v5 + limited offline numeric fallback

This project builds a record-aware ConvFinQA question-answering prototype. The assignment data already gives a selected `record_id`, so I treated the main problem as grounded reasoning inside one financial record rather than open-ended retrieval across the whole dataset.

The implementation evolved in three steps:

| Version | Idea | Purpose |
| --- | --- | --- |
| `v1` | Full-record baseline | Establish an end-to-end system using `pre_text`, table, `post_text`, and conversation history. |
| `v2` | Record-local evidence selection | Add lightweight snippet selection inside the same record to reduce wrong-row and wrong-value errors. |
| `v3` | Evidence selection plus no-gold verification retry | Add a small local verifier for suspicious final-answer format, refusal, denominator, percentage-scale, and simple calculation consistency issues. |

Version dependency and LLM-call summary:

| Version | Built from | What it does | LLM calls per turn |
| --- | --- | --- | --- |
| `v1` | Baseline | Uses the full selected record plus conversation history to answer. | `1`: final answer |
| `v2` | `v1` | Adds record-local evidence selection before answering, then still answers with evidence plus the full record. | Usually `2`: evidence reranking + final answer |
| `v3` | `v2` | Adds one no-gold verification step after the draft answer, then retries once only if the draft looks suspicious. | Usually `2`, sometimes `3`: evidence reranking + draft answer + optional retry |

The saved 500-record train results currently compare `v1` and `v2`. `v3` is the next lightweight improvement path; it does not use gold labels during answering.

## Method

Each ConvFinQA record contains:

```text
pre_text
table
post_text
dialogue history
```

The chat command loads a selected record and passes the full record context to the model. This is important because many answers are not in the table alone. Some answers come from narrative `pre_text` or `post_text`.

For example, a table may contain operating income by year, while a later follow-up asks about a decline in net earnings described only in the post-table text. Therefore the baseline prompt includes the full selected record, not only table cells.

The prompt asks the model to produce a short value check:

```text
Target:
Values:
Operation:
Final answer: <value>
Calculation:
```

The `Final answer:` line is the machine-readable answer used by the evaluator. For numeric answers, the preferred final value is a pure executable number. Percentage, portion, ratio, and rate answers should be written on the same scale as `executed_answers`, for example `0.0313` rather than `3.13%`. The evaluator still normalizes older percent-style outputs so saved experiments remain comparable.

## Evidence Selection

The second version adds record-local evidence selection. This is not full RAG: it does not search across all records, create a vector database, or use an agent framework. The selected record is already known, so the evidence problem is to highlight the most useful text sentences and table rows inside that record.

The evidence flow is:

```text
1. Generate deterministic snippets from the selected record.
2. Apply cheap lexical filtering to keep candidate snippets.
3. Ask the LLM to rerank/select the most relevant candidates.
4. Put selected snippets above the full record in the answer prompt.
5. Keep the full record as backup context.
```

The snippet generator creates three kinds of evidence:

| Snippet type | Construction rule | Why |
| --- | --- | --- |
| Text sentence | Split `pre_text` and `post_text` into readable sentences. | Narrative text often contains the needed financial values. |
| Table row | Convert each row into a compact row string with row and column labels. | A single table cell is often ambiguous without labels. |
| Number-centered clause | Split long multi-number sentences when the clause still preserves meaning. | Long financial sentences may contain competing values. |

A good snippet preserves:

```text
number + label/meaning + period/date + unit
```

For example:

```text
Bad snippet:
4.7

Good snippet:
$4.7 million of letters of credit outstanding under the credit facility
```

The reranker receives the current question, recent Q/A history, and candidate snippets, but it can only select from the candidate snippets. Conversation history is used to resolve follow-ups such as "that amount" or "during that period"; it is not treated as the source of truth.

## Evaluation Standard

The headline metric is strict execution-style accuracy against `executed_answers`.

`conv_answers` are saved for inspection and display-format debugging, but they are not used as hidden gold labels for strict correctness.

Each conversation turn is one evaluated example:

```text
turn-level accuracy = correct turns / total turns
```

The model receives previous model answers as history during batch runs, not gold previous answers. This means an early wrong answer can propagate into later turns, which matches the real chat setting.

The evaluator parses and normalizes model output before comparison:

| Parser behavior | Example |
| --- | --- |
| Prefer `Final answer:` line | Avoids grabbing dates or operands from explanations. |
| Remove commas | `25,587` -> `25587` |
| Support leading decimals | `.0751`, `-.0751` |
| Normalize explicit percentages | `5.3%` -> `0.053` |
| Infer missing percent signs when clearly supported | `Final answer: 5.3` plus `Calculation: ... = 5.3%` -> `0.053` |
| Fallback to string comparison | Supports rare `yes`/`no` executed answers. |

Numeric comparison uses:

```text
math.isclose(prediction, executed_answer, rel_tol=1e-3, abs_tol=1e-3)
```

This tolerance is my prototype choice to avoid false negatives from rounded natural-language outputs. I do not claim this tolerance is specified by the original ConvFinQA paper.

The result summaries follow the paper-style categories:

| Category | How it is derived |
| --- | --- |
| Number selection vs program question | Derived from `turn_program` for analysis only. |
| Simple vs hybrid conversation | Derived from dataset features. |
| Hybrid first/second part | Derived from `qa_split`. |
| Turn index | Turn 0, Turn 1, etc. |

`turn_program` is used only for these breakdown labels. It is not used to decide whether a model answer is correct.

## Results

The 500-record comparison used:

```text
Dataset split: train
Sample: 500 randomly sampled records
Random seed: 42
Model: gpt-4o-mini
Metric: strict execution accuracy against executed_answers
```

Saved files:

```text
outputs/1st version/run_train_500_random42_gpt4o_mini_v1.jsonl
outputs/1st version/eval_train_500_random42_gpt4o_mini_v1_strict_executed.jsonl

outputs/2nd version/run_train_500_random42_gpt4o_mini_v2.jsonl
outputs/2nd version/eval_train_500_random42_gpt4o_mini_v2_strict_executed.jsonl
```

### Headline

| Version | Method | Correct / total | Accuracy |
| --- | --- | ---: | ---: |
| `v1` | Full-record baseline | 1,223 / 1,827 | 66.9% |
| `v2` | Record-local evidence selection | 1,231 / 1,827 | 67.4% |

`v2` improves the headline result by 8 turns, or about 0.4 percentage points. The gain is real but modest, and the detailed examples show both fixes and regressions.

### Table 4-Style Breakdown

| Breakdown | v1 | v2 | Change |
| --- | ---: | ---: | ---: |
| Full results | 1,223 / 1,827 (66.9%) | 1,231 / 1,827 (67.4%) | +8 (+0.4 pp) |
| Number selection questions | 505 / 640 (78.9%) | 514 / 640 (80.3%) | +9 (+1.4 pp) |
| Program questions | 718 / 1,187 (60.5%) | 717 / 1,187 (60.4%) | -1 (-0.1 pp) |
| Simple conversations | 782 / 1,163 (67.2%) | 798 / 1,163 (68.6%) | +16 (+1.4 pp) |
| Hybrid conversations | 441 / 664 (66.4%) | 433 / 664 (65.2%) | -8 (-1.2 pp) |
| Hybrid first part | 258 / 363 (71.1%) | 248 / 363 (68.3%) | -10 (-2.8 pp) |
| Hybrid second part | 183 / 301 (60.8%) | 185 / 301 (61.5%) | +2 (+0.7 pp) |

### Turn Breakdown

| Turn | v1 | v2 | Change |
| --- | ---: | ---: | ---: |
| Turn 0 | 385 / 500 (77.0%) | 389 / 500 (77.8%) | +4 |
| Turn 1 | 349 / 500 (69.8%) | 348 / 500 (69.6%) | -1 |
| Turn 2 | 238 / 376 (63.3%) | 240 / 376 (63.8%) | +2 |
| Turn 3 | 153 / 266 (57.5%) | 146 / 266 (54.9%) | -7 |
| Turn 4 | 67 / 132 (50.8%) | 70 / 132 (53.0%) | +3 |
| Turn 5 | 20 / 36 (55.6%) | 25 / 36 (69.4%) | +5 |
| Turn 6 | 8 / 12 (66.7%) | 9 / 12 (75.0%) | +1 |
| Turn 7 | 3 / 4 (75.0%) | 4 / 4 (100.0%) | +1 |
| Turn 8 | 0 / 1 (0.0%) | 0 / 1 (0.0%) | +0 |

Later turns are harder and noisier. The final few turn indices have very small sample sizes, so their percentages should not be overinterpreted.

## Error Analysis

The original 500-record failure analysis produced these error buckets:

| Error type | Count | Proposed lightweight solution |
| --- | ---: | --- |
| Calculation/program errors | 309 | Add local calculation verification for simple arithmetic, sign, direction, and final-answer mismatch. |
| Ratio or percentage errors | 231 | Normalize percentage/ratio scale and check denominator direction. |
| Number-selection errors | 110 | Add record-local evidence selection before answering. |
| Non-numeric/refusal errors | 65 | Retry once when the model refuses despite numeric candidates in the record. |
| Other | 2 | Document annotation/task-format ambiguity rather than overfitting. |

These counts are failure-analysis labels, not the scoring metric itself. Correctness is still only decided by comparing the parsed model answer with `executed_answers`.

### Example 1: Calculation Direction

Record: `Single_ABMD/2006/page_62.pdf-1`

The first two turns are correct:

```text
Q: what was the balance at the beginning of the year in 2005?
Gold: 245
Answer: 245

Q: what was the value at the end of the year in 2006?
Gold: 167
Answer: 167
```

The third turn asks:

```text
Q: what was the difference?
Gold program: subtract(245, 167)
Gold executed_answer: 78
```

Both v1 and v2 answer:

```text
Final answer: -78
Calculation: 167 - 245 = -78
```

This is not an evidence problem. The right numbers were found, but the operation direction was reversed. The next turn then multiplies the wrong signed value by 1000, giving `-78000` instead of `78000`.

### Example 2: Evidence Selection Win

Record: `Double_C/2008/page_217.pdf`

Question:

```text
what portion of total maximum potential amount of future payments is related to financial standby letters of credit?
```

Gold:

```text
94.2 / 304.9 = 0.30895
```

v1 used the wrong partial values and answered about `26%`, which is wrong. v2 selected the correct numerator and denominator and answered about `30.9%`, which the strict evaluator normalizes to the executable ratio scale and marks correct.

This is the clearest evidence-selection success case.

### Example 3: Evidence Selection Regression

Record: `Double_ETR/2016/page_424.pdf`

Relevant post text:

```text
Entergy Texas has a credit facility in the amount of $150 million scheduled to expire in August 2021.
The credit facility allows Entergy Texas to issue letters of credit against 50% of the borrowing capacity.
As of December 31, 2016, there were no cash borrowings and $4.7 million of letters of credit outstanding under the credit facility.
```

Question:

```text
as of december 31, 2016, what was the drawn amount from the credit facility that was set to expire in august 2021?
```

Gold:

```text
4.7
```

Both v1 and v2 answered `0`, because they selected "no cash borrowings" instead of "$4.7 million of letters of credit outstanding." This then affected the follow-up percentage question. v1 later recovered and answered `3.13%` correctly under strict parser normalization, while v2 continued the wrong zero interpretation and answered `0`.

This shows that evidence selection is not automatically better. If the selected evidence reinforces the wrong interpretation, it can make later turns worse.

### Example 4: Refusal and Final-Answer Formatting

In the same `Double_ETR/2016/page_424.pdf` conversation, the follow-up asks:

```text
and what was that credit facility?
```

Gold:

```text
150
```

Both versions mention the right `$150 million` but do not always put a clean `Final answer: 150` line. The parser can then grab another number such as `2021` or `50%`. This motivated the stricter final-answer contract and the v3 verification retry.

### Example 5: Annotation / Task-Format Mismatch

Record: `Double_ADBE/2011/page_83.pdf`

Question:

```text
which weighted average useful life, then, of these two was the highest?
```

The human-style answer is `10`, or customer contracts and relationships. However, the gold program is:

```text
greater(6, 10)
```

and the gold executed answer is:

```text
no
```

The model answer `10` is semantically reasonable, but strict execution scoring marks it wrong. I treat this as a documented dataset/task-format ambiguity rather than a failure that should drive special-case code.

## V1 vs V2 Error Example Summary

| Error example | Key turn | Gold | v1 result | v2 result | What changed |
| --- | ---: | --- | --- | --- | --- |
| `Single_ABMD/2006/page_62.pdf-1` calculation direction | Turn 2 | `78` | `-78`, wrong | `-78`, wrong | No improvement. Both used the right numbers but reversed subtraction. |
| `Single_ABMD/2006/page_62.pdf-1` propagated calculation | Turn 3 | `78000` | `-78000`, wrong | `-78000`, wrong | No improvement. The earlier sign error carried forward. |
| `Double_C/2008/page_217.pdf` wrong evidence/ratio | Turn 0 | `0.30895` | `26%`, wrong | `30.9%`, correct | v2 fixed this by selecting `94.2 / 304.9`. |
| `Double_C/2008/page_217.pdf` percent formatting | Turn 3 | `0.05346` | `5.4%`, correct | `5.3%`, correct | Both are correct under latest strict parser/tolerance. |
| `Double_ETR/2016/page_424.pdf` number selection | Turn 0 | `4.7` | `0`, wrong | `0`, wrong | No improvement. Both choose cash borrowings instead of letters of credit outstanding. |
| `Double_ETR/2016/page_424.pdf` credit facility value | Turn 1 | `150` | wrong parse | wrong parse | Both mention `$150 million`, but the final answer is not clean enough. |
| `Double_ETR/2016/page_424.pdf` follow-up percentage | Turn 2 | `0.03133` | `3.13%`, correct | `0%`, wrong | v2 regressed by continuing the wrong zero interpretation. |
| `Double_ETR/2016/page_424.pdf` max letters of credit | Turn 3 | `75` | `75`, correct | `75`, correct | Both are correct. |
| `Double_ADBE/2011/page_83.pdf` annotation mismatch | Turn 2 | `no` | `10`, wrong | `10`, wrong | Human-style answer is reasonable, but gold executable answer is boolean. |

## Why V3 Focuses on Verification

The v2 evidence layer mainly targets wrong evidence selection. That directly addresses part of the `110` number-selection errors and some refusal errors. It does not reliably solve the largest remaining buckets:

```text
309 calculation/program errors
231 ratio/percentage errors
```

Many of these are "found the values, used them wrong" failures. More evidence alone cannot fix reversed subtraction, wrong denominator choice, percentage scale mistakes, or a final answer that contradicts the stated operation.

The v3 idea is therefore a lightweight no-gold verifier. It checks only the model's draft answer, the current question, and selected evidence. It does not inspect `executed_answers`, `conv_answers`, or `turn_program`.

Useful checks include:

| Check | Retry trigger |
| --- | --- |
| Refusal | The model says the record lacks enough information even though the evidence contains numeric candidates. |
| Missing final answer | The response lacks a clean `Final answer:` line. |
| Dirty final answer | The final line contains units, dates, words, or multiple numbers. |
| Percentage scale | The final answer uses display percentage form instead of executable numeric ratio scale. |
| Wrong denominator | A portion-of-total question appears to divide `total / part` instead of `part / total`. |
| Credit-facility zero trap | The answer selects `cash borrowings = 0` while evidence contains outstanding letters of credit. |
| Calculation consistency | A simple stated operation does not match the stated final value. |

For the `Double_ETR/2016/page_424.pdf` example, v3 can improve without cheating:

| Turn | Target | No-gold v3 behavior |
| ---: | --- | --- |
| 0 | Drawn amount from the August 2021 credit facility | If the draft selects `cash borrowings = 0`, retry because evidence also contains non-zero letters of credit outstanding. |
| 1 | Amount of that credit facility | Retry if the draft mentions `$150 million` but fails to output clean `Final answer: 150`. |
| 2 | Percentage that amount represented | Use `4.7 / 150 = 0.0313` on executable ratio scale. |
| 3 | Maximum letters of credit | Use `150 * 50% = 75`. |

The key point is that v3 still does not use gold answers. It only rejects locally suspicious model outputs and asks the model to answer the same question again with a more targeted instruction.

## Design Choices

I intentionally avoided several heavier approaches:

| Avoided approach | Reason |
| --- | --- |
| Full cross-record RAG | The record is already selected by `record_id`; the useful retrieval problem is inside the record. |
| Agent framework | The error analysis points to specific evidence and arithmetic issues, not a need for open-ended tool planning. |
| Full ConvFinQA DSL generation | The paper uses program generation, but a full DSL parser/executor is larger than needed for the current prototype. |
| Manual special casing of rare annotation issues | Rare `yes`/`no` mismatches should be documented rather than overfit. |

The implementation keeps the pipeline explicit:

```text
run -> save raw model predictions
evaluate -> score saved predictions against executed_answers
analyze-results -> print paper-style summaries
```

This separation is useful because parser/tolerance changes can be applied to saved model outputs without rerunning expensive API calls.

## Commands

Run v1 on the 500-record train sample:

```bash
uv run main run \
  --split train \
  --model gpt-4o-mini \
  --max-records 500 \
  --random-seed 42 \
  --version v1 \
  --workers 4 \
  --output-path "outputs/1st version/run_train_500_random42_gpt4o_mini_v1.jsonl"
```

Run v2 on the same sample:

```bash
uv run main run \
  --split train \
  --model gpt-4o-mini \
  --max-records 500 \
  --random-seed 42 \
  --version v2 \
  --workers 4 \
  --output-path "outputs/2nd version/run_train_500_random42_gpt4o_mini_v2.jsonl"
```

Evaluate a saved run:

```bash
uv run main evaluate \
  "outputs/1st version/run_train_500_random42_gpt4o_mini_v1.jsonl" \
  --output-path "outputs/1st version/eval_train_500_random42_gpt4o_mini_v1_strict_executed.jsonl"
```

Analyze a scored result:

```bash
uv run main analyze-results \
  "outputs/1st version/eval_train_500_random42_gpt4o_mini_v1_strict_executed.jsonl"
```

## Future Work

The next best engineering step is not simply "more evidence." The strongest remaining opportunity is deterministic calculation and verification support:

1. Keep v2 evidence selection for wrong-row and wrong-value cases.
2. Add v3 no-gold verification for refusals, dirty final answers, denominator mistakes, and simple operation/final-answer mismatch.
3. Add small deterministic helpers for common operations such as difference, percentage change, and part-over-total.
4. Consider full DSL/program generation only if the lightweight verifier shows a clear ceiling.

This keeps the solution close to the assignment goal: a useful, explainable prototype with measured improvements, rather than a large system whose complexity is not justified by the observed errors.

## Use of AI Assistance

I used Codex as a coding and analysis assistant while developing this prototype and report. The assistant helped inspect the repository, propose implementation changes, generate code edits, run local tests and analysis commands, and summarize failure cases. I reviewed the design decisions, commands, and outputs during the process, and the final implementation remains grounded in the saved code, dataset, and evaluation artifacts in this repository.

## Recovered Notes / Full Process Dump

This section intentionally dumps extra material from the development process. It is less polished than the main report, but it preserves decisions, observations, and explanations that were useful during the work.

### Original Paper Alignment

The ConvFinQA paper evaluates generated programs and their executed results. The paper's program-generation setup avoids most answer-text parsing issues because the model produces a DSL-like program and the evaluator executes it.

This prototype does not generate full ConvFinQA DSL programs. Instead, it is closer to an answer-only system:

```text
model answer text -> parse final answer -> compare with executed_answers
```

Because of that, the prototype needs a defensive parser. The headline metric should still be aligned with execution accuracy:

```text
strict execution accuracy = parsed model answer compared with executed_answers
```

`conv_answers` are useful for human-readable diagnostics, but they should not be used as the gold label for headline accuracy. This was an important correction during the process.

### Strict vs Display-Normalized Scoring

There are two useful views:

```text
Strict execution accuracy:
model answer -> parsed value -> compare with executed_answers only

Display-normalized chatbot accuracy:
model answer -> parsed value -> compare with display-normalized conv_answers/executed_answers
```

The strict score is the paper-aligned headline metric. The display-normalized score is only diagnostic.

Example where `conv_answers` and `executed_answers` differ:

```text
conv_answer: -3.3%
executed_answer: -0.03264
```

The executed answer is the raw ratio. A human-facing answer may use percent display, but strict scoring should compare with the executable value.

### Percentage and Ratio Handling

A major source of false negatives was percentage display. For example:

```text
Question: what portion us related to performance guarantees?
Gold executed_answer: 0.05346

Model:
Final answer: 5.3
Calculation: (16.3 / 304.9) * 100 = 5.3%
```

Without context, `5.3` looks like a plain number. But the question and calculation clearly show a percentage. The evaluator now treats this as `0.053`, then compares against `0.05346`.

This does not use `conv_answers` as gold. It only interprets the model's own output before comparing with `executed_answers`.

The current preferred output contract is even cleaner:

```text
For numerical final answers, output pure executable numbers.
For percent/portion/ratio/rate answers, output ratio scale:
0.0313, not 3.13%
```

The evaluator still accepts older percent-style outputs so old saved runs can be scored fairly.

### Answer Format Risks Found In Train

The train split contains several display-format risks:

| Format / risk type | Train count | Why it matters | Current solution |
| --- | ---: | --- | --- |
| Percentage display in `conv_answers` | 2,661 | User-facing answers often use `%`, while `executed_answers` may store raw ratios. | Normalize older percent-style model outputs to ratio scale for comparison; v3 prompts numeric ratio-scale final answers. |
| Percent display where executed value is ratio-scale | 2,406 | Example: `14.1%` vs `0.14136`; raw string comparison would fail. | Compare normalized numeric values, not raw strings. |
| Percent display with executed value larger than 1 | 255 | Large percentages such as `700%` execute to `7.0`; this is still valid ratio-scale. | Same percentage normalization handles this without special casing. |
| Plain display with small decimal executed answer | 466 | Some decimal answers are valid ratios, while others may be percentage-like displays without `%`. | Keep as decimal unless the question/calculation clearly implies percent. |
| Leading decimal answers | 112 | Values such as `.0751` or `-.62` can be misread by naive regex. | Parser regex supports leading decimals. |
| Old regex differs from new regex | 112 | Confirms every detected leading-decimal case was at risk. | Fixed by the leading-decimal parser update. |
| Non-numeric / no parseable number | 38 | Some answers are `yes`, `no`, `ye`, or empty strings. | Fall back to normalized string comparison when numeric parsing fails. |
| Parenthesized number | 1 | Parentheses can indicate a displayed negative or contextual value. | Rare in train; keep under observation. |

Related dataset check: `executed_answers` are almost all numeric, but not literally all numeric. A scan found `12,554` numeric answers and `40` string answers such as `yes`/`no`. Therefore pure numeric final answers are preferred for numeric cases, but the evaluator keeps a string fallback.

### Why The Parser Uses Final Answer

Early model answers sometimes included several numbers:

```text
The credit facility is in the amount of $150 million scheduled to expire in August 2021.
It allows letters of credit against 50% of the facility.
```

If the parser grabs the wrong number, it may compare `2021` or `50` instead of `150`.

The prompt therefore requires:

```text
Final answer: 150
```

The parser prefers that line. If it is missing, it falls back to the last numeric value, but this fallback is less reliable and motivates v3 verification.

### Run / Evaluate Separation

I separated model calls from evaluation:

```text
run -> save raw model predictions
evaluate -> score saved predictions
analyze-results -> summarize scored predictions
```

This matters because:

- model calls are slow and cost money
- parser and tolerance changes can be reapplied offline
- v1 and v2 can be compared fairly using the same evaluation standard
- raw prediction files do not store correctness flags or gold answers

Raw run JSONL stores:

```text
record_id
turn_index
question
prediction
```

Evaluation JSONL stores:

```text
record_id
turn_index
question
prediction
prediction_value
prediction_is_percent
gold_conv_answer
gold_executed_answer
gold_value
gold_is_percent
is_correct
```

### Parallel Running

Batch runs support `--workers`. Parallelism is per record, and result rows are combined in dataset order, so output JSONL remains compatible with evaluation.

For 500 records, `--workers 4` was a practical balance. Higher worker counts such as 10 can hit OpenAI rate limits faster. Retry helps with temporary token-per-minute or request-per-minute pressure, but it cannot solve a hard requests-per-day limit.

If a run completes with retries, the result is still valid. A retry just means the same requested model call succeeded later.

### Rate Limits Observed

One failure hit token-per-minute:

```text
Rate limit reached for gpt-4o-mini on tokens per min (TPM)
```

This can usually be handled with retry/backoff or fewer workers.

Another failure hit requests-per-day:

```text
Rate limit reached for gpt-4o-mini on requests per day (RPD): Limit 10000, Used 10000
```

This cannot be solved by retrying immediately. It requires waiting for the limit reset or using fewer total calls.

### Why `max_records` Defaults To All

Originally `max_records` defaulted to a small number for smoke testing. That was confusing for full train runs. The better default is:

```text
max_records = None
```

This means run all records unless the user explicitly sets a limit:

```bash
--max-records 500
```

or:

```bash
--max-records 10000
```

Using `10000` covers all train records if the train split has fewer records than that, but a default of `None` is cleaner.

### Evidence Construction Details

Snippet generation is deterministic and separate from ranking.

Text snippets:

```text
Split pre_text and post_text into sentences.
Keep the full sentence where possible.
```

Table snippets:

```text
Convert each table row to a compact text row with row label, column labels, and values.
```

Number-centered clauses:

```text
For long sentences with many numbers, create smaller clauses only when they still preserve meaning.
```

Bad snippet:

```text
1379
```

Good table-row snippet:

```text
total shares purchased as part of publicly announced programs |
October: 2506 | November: 1923 | December: 1379
```

Bad snippet:

```text
4.7
```

Good text snippet:

```text
$4.7 million of letters of credit outstanding under the credit facility
```

The reranker is not asked to invent evidence. It selects from deterministic candidates.

### V2 Uses An Additional API Call

`v2` uses the same selected model, normally `gpt-4o-mini`, for evidence reranking. This means a v2 answer generally needs:

```text
1 API call for evidence reranking
1 API call for final answering
```

`v1` normally needs only the final answering call.

`v3` can need an additional final-answer retry call if the verifier flags a suspicious draft.

### Example Of The Evidence Selection Call

The reranker sees something like:

```text
Current question:
what percentage of the facility was used?

Conversation history:
Q: what was the outstanding amount of letters of credit?
A: Final answer: 4.7

Candidate snippets:
[T-31] Entergy Texas has a credit facility in the amount of $150 million scheduled to expire in August 2021.
[T-33] As of December 31, 2016, there were no cash borrowings and $4.7 million of letters of credit outstanding under the credit facility.
[T-32] The credit facility allows Entergy Texas to issue letters of credit against 50% of the borrowing capacity of the facility.
```

Expected selected evidence:

```text
[T-31] facility amount = 150
[T-33] letters of credit outstanding = 4.7
```

Then the final answer model can compute:

```text
4.7 / 150 = 0.0313
```

### Why Evidence Selection Helped Only Modestly

The v2 result is slightly better overall:

```text
v1: 1223 / 1827 = 66.9%
v2: 1231 / 1827 = 67.4%
```

This is a modest improvement because evidence selection only targets part of the problem. It helps when the main failure is choosing the wrong row, sentence, numerator, or denominator. It does not directly solve:

- reversed subtraction
- wrong operation direction
- final answer not matching operation
- percentage scale mistakes
- refusals where the answer exists
- annotation/task-format mismatches

This is why v3 focuses on verification rather than simply adding more evidence.

### Why Calculation Verification Is The Next Step

The largest failure buckets are:

```text
Calculation/program errors: 309
Ratio/percentage errors: 231
```

Together these are `540` error labels. Many are not "could not find the evidence" failures. They are "found the values, used them wrong" failures.

Examples:

```text
Found 245 and 167, but computed 167 - 245 instead of 245 - 167.
Found part and total, but divided total / part instead of part / total.
Computed a percentage, but final-answer scale did not match executed_answers.
Mentioned the right 150 but final answer line contained extra context.
```

A lightweight verifier can catch some of these without gold answers.

### V3 No-Gold Verification

The verifier does not use:

```text
executed_answers
conv_answers
turn_program
```

It checks only:

```text
question
selected evidence
model Target/Values/Operation/Final answer/Calculation
```

Current/recommended checks:

| Check | Purpose |
| --- | --- |
| Missing `Final answer:` | Avoid parser grabbing random numbers. |
| Dirty final answer | Force one comparable value only. |
| Refusal with numeric evidence | Retry because ConvFinQA turns usually have an answer. |
| Credit-facility zero trap | Avoid choosing `cash borrowings = 0` when letters of credit outstanding better matches the question. |
| Portion-of-total denominator | Prefer `part / total`, not `total / part`. |
| Percentage scale | Prefer executable ratio scale, not display percent form. |
| Operation/final mismatch | If operation says `245 - 167` but final is `-78`, retry. |

The verifier is conservative. If it is unsure, it should accept the answer rather than overcorrect.

### V3 On The Entergy Example

Record:

```text
Double_ETR/2016/page_424.pdf
```

Useful evidence:

```text
Entergy Texas has a credit facility in the amount of $150 million scheduled to expire in August 2021.
The credit facility allows Entergy Texas to issue letters of credit against 50% of the borrowing capacity.
As of December 31, 2016, there were no cash borrowings and $4.7 million of letters of credit outstanding under the credit facility.
```

Expected strict answers:

| Turn | Question target | Expected |
| ---: | --- | ---: |
| 0 | drawn amount from facility | 4.7 |
| 1 | amount of that credit facility | 150 |
| 2 | percentage/portion that amount represented | 0.03133 |
| 3 | max letters of credit that could be issued | 75 |

Why v3 can fix this without cheating:

- Turn 0: if the draft says `cash borrowings = 0`, the verifier sees evidence also contains non-zero letters of credit outstanding.
- Turn 1: if the draft says a sentence like `$150 million scheduled to expire in 2021`, the verifier asks for clean `Final answer: 150`.
- Turn 2: the carried values should be `4.7` and `150`; final answer should be executable ratio `0.0313`.
- Turn 3: use the 50% rule: `150 * 0.5 = 75`.

### V3 Does Not Cheat

V3 can get some examples correct without using gold because its checks come from local consistency:

```text
Question asks for current credit facility amount.
Evidence contains 150 and 2021.
Draft final answer includes both 150 and 2021.
Verifier says final answer line has multiple numbers.
Retry asks for exactly one comparable value.
```

This does not require knowing the gold answer is `150`; it only knows the final line is not machine-clean.

### Few-Shot Prompt Experiment

I considered adding a few-shot prompt to v3 after discussing it as a possible improvement. The idea was to include compact pattern examples for:

- credit facility sentences with competing values
- part / total percentage questions
- clean final-answer format

However, this was later removed from the final v3 direction because it could influence examples too strongly and made the code/prompt less clean. The current preferred v3 definition is:

```text
v3 = evidence selection + no-gold verification retry
```

No few-shot examples are needed in the final implementation.

### Full DSL / Program Generation

The ConvFinQA paper uses program generation. A full DSL approach would mean generating expressions such as:

```text
subtract(245, 167)
divide(94.2, 304.9)
multiply(150, 50%)
```

and then executing them. This can be powerful because it separates reasoning from final answer formatting.

I did not implement full DSL generation because:

- it is a larger system
- it requires robust program parsing/execution
- the assignment rewards optimal scope
- the current failure analysis suggests smaller targeted fixes first

A middle-ground future version could generate a lightweight calculation plan without implementing the full ConvFinQA DSL.

### Bayesian-Lite Idea

I considered a Bayesian framing because the paper's latent-program formulation naturally suggests scoring possible evidence, operations, and answers.

For this assignment, that idea is best kept as a future extension:

```text
score candidate evidence snippets
score candidate operation plans
score candidate final answers
choose answer with best lightweight confidence
```

This should not become full Bayesian inference or MCMC. It would be a practical scoring layer over candidate plans.

### Error Type Table: Evidence Helps Or Not

| Error type | Error count in 500-record analysis | Does evidence selection help? | Why |
| --- | ---: | --- | --- |
| Number-selection / wrong evidence | 110 | Yes, directly | LLM reranking can surface the right table row or text sentence before the answer step. |
| Non-numeric/refusal where answer exists | 65 | Yes, partly | If the model refused because it missed evidence, selected snippets can make the answer easier to find. |
| Calculation/program errors | 309 | Partly | Better evidence helps only when the model used wrong numbers. It does not fix wrong operation, sign, or direction. |
| Ratio/percentage scale mistakes | 231 | Partly / no | Evidence may clarify units or context, but parser normalization and verification are the main fixes. |
| Annotation ambiguity / dataset issue | 2 | No | Evidence cannot fix questionable gold format or ambiguous task wording. |

### V1 And V2 Are Not Separate Implementations

The versions share most of the same answering pipeline. That is intentional:

```text
v1 = full record -> answer
v2 = select evidence + full record -> answer
v3 = select evidence + full record -> answer -> verify/retry if suspicious
```

They are versioned paths through the same clean modules, not three copied codebases.

For final submission, it is acceptable to keep the final best implementation clean and document the development process in the report. It is not necessary to preserve every old implementation path forever if a later version clearly replaces it.

The report preserves:

```text
baseline
failure analysis
evidence-selection experiment
verification rationale
```

### Code Structure Notes

High-level script responsibilities:

| File | Role |
| --- | --- |
| `src/main.py` | Thin Typer CLI: parse options, load data, call answering/evaluation modules, print output. |
| `src/answers.py` | Calls OpenAI for final answers and coordinates version behavior. |
| `src/prompts.py` | Builds system/chat prompts only. |
| `src/evidence.py` | Builds and selects record-local evidence snippets. |
| `src/evaluation.py` | Parses, normalizes, compares answers, writes scored results, builds breakdowns. |
| `src/verification.py` | No-gold local verification and one retry instruction for v3. |
| `src/formatting.py` | Deterministic formatting of records/tables. |
| `src/logger.py` | Shared logger helper. |
| `src/models.py` | Pydantic data models. |

Design preference:

```text
main.py should stay thin.
answers.py should call the model.
evaluation.py should own parse/normalize/compare.
verification.py should own no-gold retry checks.
```

### Commands Used Often

Single-record chat:

```bash
uv run main chat Double_ETR/2016/page_424.pdf --version v3 --show-evidence
```

Run v1:

```bash
uv run main run \
  --split train \
  --model gpt-4o-mini \
  --max-records 500 \
  --random-seed 42 \
  --version v1 \
  --workers 4 \
  --output-path "outputs/1st version/run_train_500_random42_gpt4o_mini_v1.jsonl"
```

Run v2:

```bash
uv run main run \
  --split train \
  --model gpt-4o-mini \
  --max-records 500 \
  --random-seed 42 \
  --version v2 \
  --workers 4 \
  --output-path "outputs/2nd version/run_train_500_random42_gpt4o_mini_v2.jsonl"
```

Evaluate:

```bash
uv run main evaluate \
  "outputs/1st version/run_train_500_random42_gpt4o_mini_v1.jsonl" \
  --output-path "outputs/1st version/eval_train_500_random42_gpt4o_mini_v1_strict_executed.jsonl"
```

Analyze:

```bash
uv run main analyze-results \
  "outputs/1st version/eval_train_500_random42_gpt4o_mini_v1_strict_executed.jsonl"
```

Full train run should omit `--max-records` if the default is all records:

```bash
uv run main run \
  --split train \
  --model gpt-4o-mini \
  --version v1 \
  --workers 4 \
  --output-path outputs/run_train_all_gpt4o_mini_v1.jsonl
```

### Result Interpretation Notes

The v2 improvement is not huge, but it is directionally useful:

```text
v1: 1223/1827
v2: 1231/1827
net gain: +8
```

The small gain is believable because v2 has both fixes and regressions:

```text
helps wrong evidence cases
can regress when evidence selection emphasizes the wrong phrase
does not fix arithmetic reasoning errors
```

The `Double_C/2008/page_217.pdf` example is a good v2 win.

The `Double_ETR/2016/page_424.pdf` example is a good v2 limitation/regression.

The `Single_ABMD/2006/page_62.pdf-1` example is a good calculation-verification motivation.

### Development Philosophy

The implementation philosophy throughout was:

```text
Add the smallest component that addresses an observed failure mode.
Measure it on train first.
Keep dev closer to a final blind evaluation.
Avoid full RAG, agents, or full DSL generation until evidence says they are worth it.
```

This is also why the report is important: even if the final code keeps only the best path, the report documents:

- baseline result
- failure analysis
- evidence-selection result
- evaluation standard
- remaining limitations
- why verification is the next component

### Known Limitations

Remaining limitations:

- v2 reranking uses an additional model call, increasing cost and rate-limit pressure.
- Selected evidence can still be misleading.
- No full symbolic executor exists yet.
- v3 verifier catches only conservative local patterns.
- Some gold labels are boolean strings even when the natural-language question sounds value-seeking.
- Full-train runs can hit OpenAI daily request limits.
- Saved train results are useful for development, but dev should be used for final validation.

### Final Submission Note

For submission, the codebase should be kept clean. It is better to submit the best final path and explain previous versions in the report than to keep messy experimental branches in the production code.

The final report should make clear:

```text
I first built a full-record baseline to establish an end-to-end system and measure errors.
After failure analysis, I added record-local evidence selection.
Since the evidence-selection version became the preferred final approach, the submitted implementation keeps the final path clean while the report documents the baseline and comparison.
```

### Extra Version Comparison Tables

I repeatedly compared v1, v2, and v3 during development. The most important distinction is:

| Version | Name | Uses evidence selection? | Uses no-gold verification retry? | Uses few-shot prompt? | Purpose |
| --- | --- | --- | --- | --- | --- |
| `v1` | Full-record baseline | No | No | No | Establish a simple end-to-end baseline using the full selected record. |
| `v2` | Evidence-selection version | Yes | No | No | Test whether record-local snippet selection improves wrong-evidence cases. |
| `v3` | Verification version | Yes | Yes | No | Keep v2 evidence selection, then retry once when the draft answer is locally suspicious. |

Important note: the few-shot prompt idea was explored but later removed from the intended v3 design. The clean v3 definition is:

```text
v3 = evidence selection + no-gold verification retry
```

The 500-record strict executed-answer result table:

| Version | Method | Correct / total | Accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| `v1` | Full-record baseline | 1,223 / 1,827 | 66.9% | Strong baseline; sees full record but can choose wrong values. |
| `v2` | Record-local evidence selection | 1,231 / 1,827 | 67.4% | Net +8 turns; helps some wrong-evidence cases but causes some regressions. |
| `v3` | Evidence + no-gold verification retry | Not fully evaluated in saved 500-record table | N/A | Designed to target remaining calculation/format/refusal issues without gold labels. |

Detailed v1 vs v2 breakdown:

| Breakdown | v1 | v2 | Difference |
| --- | ---: | ---: | ---: |
| Full results | 1,223 / 1,827 (66.9%) | 1,231 / 1,827 (67.4%) | +8 (+0.4 pp) |
| Number selection questions | 505 / 640 (78.9%) | 514 / 640 (80.3%) | +9 (+1.4 pp) |
| Program questions | 718 / 1,187 (60.5%) | 717 / 1,187 (60.4%) | -1 (-0.1 pp) |
| Simple conversations | 782 / 1,163 (67.2%) | 798 / 1,163 (68.6%) | +16 (+1.4 pp) |
| Hybrid conversations | 441 / 664 (66.4%) | 433 / 664 (65.2%) | -8 (-1.2 pp) |
| Hybrid conversations, first part | 258 / 363 (71.1%) | 248 / 363 (68.3%) | -10 (-2.8 pp) |
| Hybrid conversations, second part | 183 / 301 (60.8%) | 185 / 301 (61.5%) | +2 (+0.7 pp) |

Turn-by-turn v1 vs v2:

| Turn | v1 | v2 | Difference |
| --- | ---: | ---: | ---: |
| Turn 0 | 385 / 500 (77.0%) | 389 / 500 (77.8%) | +4 |
| Turn 1 | 349 / 500 (69.8%) | 348 / 500 (69.6%) | -1 |
| Turn 2 | 238 / 376 (63.3%) | 240 / 376 (63.8%) | +2 |
| Turn 3 | 153 / 266 (57.5%) | 146 / 266 (54.9%) | -7 |
| Turn 4 | 67 / 132 (50.8%) | 70 / 132 (53.0%) | +3 |
| Turn 5 | 20 / 36 (55.6%) | 25 / 36 (69.4%) | +5 |
| Turn 6 | 8 / 12 (66.7%) | 9 / 12 (75.0%) | +1 |
| Turn 7 | 3 / 4 (75.0%) | 4 / 4 (100.0%) | +1 |
| Turn 8 | 0 / 1 (0.0%) | 0 / 1 (0.0%) | +0 |

Representative error-example comparison:

| Error example | Key turn | Gold | v1 result | v2 result | v3 expected behavior |
| --- | ---: | --- | --- | --- | --- |
| `Single_ABMD/2006/page_62.pdf-1` calculation direction | Turn 2 | `78` | `-78`, wrong | `-78`, wrong | Verification can flag operation/final mismatch if stated operation and final value disagree; direction itself remains hard if wording is ambiguous. |
| `Single_ABMD/2006/page_62.pdf-1` propagated calculation | Turn 3 | `78000` | `-78000`, wrong | `-78000`, wrong | If turn 2 is fixed, propagation improves; otherwise this remains wrong. |
| `Double_C/2008/page_217.pdf` wrong evidence/ratio | Turn 0 | `0.30895` | `26%`, wrong | `30.9%`, correct | v2 already fixes this; v3 should preserve the win and prefer executable ratio scale. |
| `Double_C/2008/page_217.pdf` percent formatting | Turn 3 | `0.05346` | `5.4%`, correct | `5.3%`, correct | Evaluator normalizes old percent output; v3 should output a ratio such as `0.0534`. |
| `Double_ETR/2016/page_424.pdf` number selection | Turn 0 | `4.7` | `0`, wrong | `0`, wrong | Credit-facility zero-trap verifier can retry when draft selects `cash borrowings = 0` despite outstanding letters of credit. |
| `Double_ETR/2016/page_424.pdf` credit facility value | Turn 1 | `150` | wrong parse | wrong parse | Dirty-final-answer verifier can retry until the final line is `Final answer: 150`. |
| `Double_ETR/2016/page_424.pdf` follow-up percentage | Turn 2 | `0.03133` | `3.13%`, correct | `0%`, wrong | If turn 0 is corrected to `4.7`, v3 should compute `4.7 / 150 = 0.0313`. |
| `Double_ETR/2016/page_424.pdf` max letters of credit | Turn 3 | `75` | `75`, correct | `75`, correct | Should remain correct. |
| `Double_ADBE/2011/page_83.pdf` annotation/task-format mismatch | Turn 2 | `no` | `10`, wrong | `10`, wrong | Not a good target for special logic; document as dataset/task-format ambiguity. |

Short interpretation:

```text
v1 is simple and surprisingly strong.
v2 improves the specific wrong-evidence problem but has regressions.
v3 is motivated because many remaining errors are not evidence-location errors; they are verification, arithmetic, denominator, and output-format errors.
```

### Development Progress Rationale: Why v1 -> v2 -> v3

The development path was deliberately incremental. I did not start with a complex system because the assignment asks for an effective prototype and good engineering judgement, not maximum architecture.

#### Why v1 first

The first goal was to make the starter app actually solve the core task end to end. The original chat interface could receive a record ID, but the model needed the selected financial document content to answer correctly.

So v1 focused on the smallest complete baseline:

```text
selected record
-> pre_text + table + post_text
-> conversation history
-> current question
-> model final answer
-> parse/evaluate against executed_answers
```

This baseline was necessary for three reasons:

1. It proved the full pipeline worked: chat, batch run, saved outputs, offline evaluation, and summary tables.
2. It created a fair measurement point before adding extra components.
3. It produced real failure examples instead of guessing what to optimize.

The v1 result was:

```text
1,223 / 1,827 = 66.9%
```

That was strong enough to show the baseline was meaningful, but the failure analysis showed clear remaining problems.

#### What v1 taught us

The v1 error analysis showed five main buckets:

```text
Calculation/program errors: 309
Ratio/percentage errors: 231
Number-selection errors: 110
Non-numeric/refusal errors: 65
Other: 2
```

This told us that not all errors had the same cause. Some were wrong evidence, some were wrong calculation, some were output-format/parser issues, and some were dataset ambiguities.

The important lesson was:

```text
Do not blindly build a full RAG system or full program generator.
First add the smallest component that targets an observed failure mode.
```

#### Why v2 next

The first targeted improvement was record-local evidence selection.

Reason:

```text
Number-selection errors and some refusals looked like "the answer exists in the record, but the model focused on the wrong sentence/table row."
```

Examples:

- choosing a wrong numerator or denominator
- missing a value in `post_text`
- focusing on "no cash borrowings" instead of "$4.7 million letters of credit outstanding"
- refusing even though the record contains the needed value

Since the record ID is already given, the problem is not cross-document retrieval. A full vector database would be unnecessary. The lightweight v2 design was:

```text
generate snippets inside selected record
-> lexical candidate filtering
-> LLM reranking of candidate snippets
-> answer with selected evidence plus full record backup
```

This is why v2 is "record-local evidence selection," not full RAG.

The v2 result was:

```text
1,231 / 1,827 = 67.4%
```

That is a net gain of 8 turns over v1.

#### What v2 taught us

v2 helped in exactly the kind of case it was designed for. For `Double_C/2008/page_217.pdf`, v1 used wrong partial values, while v2 selected the correct numerator and denominator:

```text
94.2 / 304.9 = 0.30895
```

However, v2 also regressed in some follow-up cases. In `Double_ETR/2016/page_424.pdf`, both v1 and v2 initially selected `cash borrowings = 0` instead of `letters of credit outstanding = 4.7`. v2 then continued that wrong interpretation into the percentage follow-up.

This showed:

```text
Evidence selection is useful but not sufficient.
Selected evidence can still be misinterpreted.
Better evidence does not automatically fix wrong operations or dirty final answers.
```

The v2 result was therefore honest: a modest improvement, not a complete solution.

#### Why v3 after v2

After v2, the biggest remaining issue was no longer only finding evidence. The largest buckets were still:

```text
Calculation/program errors: 309
Ratio/percentage errors: 231
```

These are mostly reasoning and verification problems:

- right values but wrong subtraction direction
- right values but wrong denominator
- final answer does not match the stated operation
- output uses display percent scale instead of executable ratio scale
- answer contains the right value but also dates/units/extra numbers
- model refuses even though selected evidence has numeric candidates

This motivated v3:

```text
v3 = v2 evidence selection + no-gold verification retry
```

The key constraint is "no-gold." In real use, the system does not know `executed_answers`, `conv_answers`, or `turn_program`. So v3 can only check local consistency:

```text
question
selected evidence
model's Target/Values/Operation/Final answer/Calculation
```

If a conservative check fires, v3 retries once with a targeted correction instruction.

#### Why not full DSL immediately

The ConvFinQA paper uses program generation, and full DSL generation could be valuable. But it would add a lot:

- generate valid programs
- parse operations
- execute operations
- handle constants and references
- compare program accuracy
- debug another failure surface

Given the assignment scope, a lightweight verifier is a better next step because it targets the same broad issue, calculation reliability, with much less complexity.

The intended path is:

```text
v1: make the system work end to end
v2: improve evidence grounding
v3: improve local answer/calculation reliability
future: consider full DSL only if lightweight verification is not enough
```

#### Why not keep every version in final code

During development, versions are useful for comparison. For final submission, the code should stay clean. The report can document:

```text
v1 baseline
v2 evidence experiment
v3 verification rationale
```

while the implementation keeps only the clean versioned paths that are still useful.

The engineering principle is:

```text
Use experiments to learn.
Use the report to preserve the learning.
Keep the submitted codebase clean.
```

## Train/Dev Leakage Audit For Possible v4 Retrieval

We discussed a possible v4 idea:

```text
v4 = v3 + retrieved similar train examples as few-shot guidance
```

The important condition is that retrieval can only use train data. It must not retrieve dev/test gold answers.

I ran a quick local leakage audit between train and dev before deciding whether this is a reasonable future direction.

Audit results:

| Leakage level | Result | Interpretation |
| --- | ---: | --- |
| Exact `record_id` overlap | `0` | Good. No same record IDs appear in both train and dev. |
| Exact document/context hash overlap | `0` | Good. No exact same `pre_text + table + post_text` record content appears in both splits. |
| Same base PDF/page overlap | `0` | Good. No same `.../page_x.pdf` appears in both train and dev. |
| Same company-year overlap | `144` company-year pairs | Related companies/years appear across different pages. This is not direct leakage, but train and dev are not fully independent by issuer/year. |
| Exact question text overlap | `180` question strings | Significant template reuse. Many are generic follow-ups such as `and in 2013?`. |
| Exact question + executed answer overlap | `5` | Very small. Mostly generic questions with coincidentally same numeric answers. |
| Normalized question signature overlap | `161` | Strong operation/template overlap after normalizing years and numbers. |
| Dev turns with near train question Jaccard >= `0.85` | `506 / 1490` | Many dev questions have near-identical train wording patterns. |

Judgement:

```text
No clear data leakage at the document or answer level.
But there is strong question-template overlap between train and dev.
```

This means retrieved train examples could be useful for v4 because many dev questions share reasoning patterns with train questions, such as:

- percentage change
- part / total ratio
- difference between years
- follow-up references such as `that amount`
- clean final-answer formatting

However, the method needs strict guardrails:

```text
Retrieve only from train.
Retrieve by question/pattern similarity, not by answer.
Use examples as reasoning-format guidance only.
Current answer must still come from the current record.
Never copy numbers from retrieved examples.
Do not retrieve from dev/test labels.
```

Fair judgement:

```text
v4 train-example retrieval is feasible and potentially beneficial.
It is not direct leakage if implemented carefully.
It should be treated as a future extension or stretch experiment after v3 evaluation.
```

Why v4 may help:

```text
ConvFinQA has repeated reasoning templates.
The leakage audit found strong train/dev question-template overlap without exact document overlap.
This means train examples can teach operation patterns without giving away the current dev answer.
```

Useful patterns that retrieved train examples may help with:

| Pattern | Why retrieved examples may help |
| --- | --- |
| Part / total ratios | Similar examples can reinforce using `part / total`, not `total / part`. |
| Percentage change | Similar examples can show whether the denominator should be the earlier value, later value, or referenced total. |
| Difference direction | Similar examples can show the expected order for "change from X to Y" or "difference between these values." |
| Follow-up references | Similar examples can show how to resolve `that amount`, `this value`, or `during that period` from previous turns. |
| Clean final-answer format | Examples can reinforce `Target`, `Values`, `Operation`, and `Final answer:` structure. |
| Repeated financial phrasing | Similar examples can help interpret recurring terms such as credit facility, outstanding, carrying value, fair value, and amortization. |

Why this is different from leaking answers:

```text
The retrieved train examples provide reasoning patterns.
The current answer still has to be computed from the current record's evidence.
The prompt should explicitly say not to copy numbers from examples.
```

Risk:

```text
Retrieved examples can distract the model if they are superficially similar but require a different operation.
Including gold answers from train is acceptable as few-shot demonstration, but must be documented clearly.
The prompt must explicitly say examples are patterns only and all current numbers must come from the current record.
```

Dataset-size check for current-record retrieval:

```text
Measured over train + dev records after formatting each record as prompt context.
records: 3458
```

| Metric | Words per formatted record |
| --- | ---: |
| min | 95 |
| mean | 707 |
| median | 705 |
| p90 | 998 |
| p95 | 1093 |
| p99 | 1551 |
| max | 2362 |

| Metric | Table rows |
| --- | ---: |
| median | 5 |
| p95 | 9 |
| max | 19 |

Interpretation:

```text
Each record is a single filing excerpt plus one small table, and the formatted context comfortably fits in a modern LLM context window.
Because the current record is small, embeddings/vector-store retrieval over the current document is unnecessary.
Lightweight evidence selection can still help focus the model, but full chunking/vector RAG is not needed for context length.
```

## Paper Pain Points And Our Responses

This table maps the main paper/ConvFinQA pain points to the relevant paper locations and our current response.

Source paper: `https://arxiv.org/pdf/2210.03849`

| Paper / ConvFinQA pain point | Paper reference | Our response |
| --- | --- | --- |
| Financial QA needs document grounding over text + tables | Section 3 defines the input as financial report textual content `T` plus structured table `B`, lines 160-168. | v1 loads the selected record's `pre_text`, table, and `post_text`. |
| Later questions depend on conversation history | Section 3 says later questions may depend on previous questions, lines 161-168; Figure 1 also shows each question may depend on previous questions, lines 47-61. | All versions pass previous Q/A turns as conversation history. |
| Questions include both direct number lookup and calculations | Dataset construction says users ask surface-content questions, calculation questions, and sequential combinations, lines 181-188. | Prompt asks for `Target`, `Values`, `Operation`, `Final answer`, and `Calculation`. |
| Need reasoning programs / execution accuracy | Section 3 says the target is to generate a reasoning program and evaluate execution result/program equivalence, lines 166-178. | We do not generate full DSL programs, but strict evaluation compares parsed answers against `executed_answers`. |
| Evidence retrieval matters | FinQANet retrieves supporting facts before program generation; the paper reports top-3 fact recall and concatenates retrieved facts with conversation context, lines 437-444. | v2/v3 add record-local evidence selection with snippets + LLM reranking. |
| Gold supporting facts improve performance | Table 3 includes `FinQANet-Gold` and notes using gold supporting facts, lines 431-436. | Supports our intuition that better evidence can help, though we use predicted/local evidence, not gold facts. |
| Number-selection questions are easier than program questions | Table 4 shows number-selection questions at 82.54 Exe Acc vs program questions at 62.14, lines 456-465. | Our analysis reports the same breakdown style and separates number selection vs program questions. |
| Hybrid conversations and later turns are harder | Table 4 and Figure 5 discussion say hybrid conversations, especially second part, and later turns are harder, lines 458-479. | We report simple/hybrid/turn-index breakdowns and preserve model history during runs. |
| Missing facts / wrong values / wrong math are key errors | Analysis says lack of domain knowledge leads to missing retrieval facts, wrong value selections, and wrong mathematical generations, lines 491-496. | v2 targets wrong evidence/value selection; v3 targets suspicious calculation/final-answer issues. |
| Long reasoning chains cause propagation errors | Paper says later turns with longer dependencies are difficult, and if any turn is wrong, later turns have little chance, lines 497-503. | We observed propagation in examples and added v3 retry checks, but this remains only partially solved. |
| Full report in prompt may be unrealistic for prompting methods | Section 6 says directly injecting the full financial report into GPT-3 prompt is unrealistic due to length, lines 510-519. | Our dataset records are small enough for this prototype, but v2 evidence selection is a lightweight move toward retriever-generator design. |
| Prompting/few-shot methods can mimic examples or ignore context | Paper says GPT-3 may mimic exemplars or reason from general-domain knowledge instead of actual context, lines 653-666. | This supports our decision to remove the few-shot prompt from v3 and prefer evidence + verification. |
| GPT-style models struggle with complex calculations | Paper notes GPT-3 struggles with complex calculations such as long digits and divisions, lines 637-646. | v3 adds lightweight verification; full symbolic/program execution remains future work. |
| Current dataset does not cover all real-world conversations | Limitations say their construction mechanisms do not cover all real-world cases, lines 690-696. | We do not build open-ended chat without `record_id`; cross-record retrieval is future work. |

Short conclusion:

```text
We directly address grounding, conversational history, strict executed-answer evaluation, and local evidence selection.
We partially address numerical reasoning with answer-format checks and no-gold verification.
We do not fully solve the paper's program-generation/symbolic-reasoning challenge.
```

## V1 / V2 / V3 500-Record Result Comparison

Run setting:

```text
split: train
sample: 500 records
random seed: 42
model: gpt-4o-mini
metric: strict execution accuracy against executed_answers
```

| Breakdown | v1 | v2 | v3 | v2-v1 | v3-v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full results | 1223/1827 (66.9%) | 1231/1827 (67.4%) | 1275/1827 (69.8%) | +0.4 pts | +2.4 pts |
| Number selection questions | 505/640 (78.9%) | 514/640 (80.3%) | 522/640 (81.6%) | +1.4 pts | +1.3 pts |
| Program questions | 718/1187 (60.5%) | 717/1187 (60.4%) | 753/1187 (63.4%) | -0.1 pts | +3.0 pts |
| Simple conversations | 782/1163 (67.2%) | 798/1163 (68.6%) | 819/1163 (70.4%) | +1.4 pts | +1.8 pts |
| Hybrid conversations | 441/664 (66.4%) | 433/664 (65.2%) | 456/664 (68.7%) | -1.2 pts | +3.5 pts |
| Hybrid conversations first part | 258/363 (71.1%) | 248/363 (68.3%) | 271/363 (74.7%) | -2.8 pts | +6.3 pts |
| Hybrid conversations second part | 183/301 (60.8%) | 185/301 (61.5%) | 185/301 (61.5%) | +0.7 pts | +0.0 pts |
| Turn 0 | 385/500 (77.0%) | 389/500 (77.8%) | 392/500 (78.4%) | +0.8 pts | +0.6 pts |
| Turn 1 | 349/500 (69.8%) | 348/500 (69.6%) | 358/500 (71.6%) | -0.2 pts | +2.0 pts |
| Turn 2 | 238/376 (63.3%) | 240/376 (63.8%) | 250/376 (66.5%) | +0.5 pts | +2.7 pts |
| Turn 3 | 153/266 (57.5%) | 146/266 (54.9%) | 165/266 (62.0%) | -2.6 pts | +7.1 pts |
| Turn 4 | 67/132 (50.8%) | 70/132 (53.0%) | 76/132 (57.6%) | +2.3 pts | +4.5 pts |

Main reading:

```text
v2 gives a small overall gain over v1, mostly from evidence helping number selection.
v3 is the clearer jump: +2.4 points over v2 overall.
The v3 gain is strongest on program questions, hybrid first-part questions, and deeper turns.
```

### V4 Train-500 Update

After completing the fair v4 train-500 run with train-sample leakage controls:

| Breakdown | v1 | v2 | v3 | v4 | v2-v1 | v3-v2 | v4-v3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full results | 1223/1827 (66.9%) | 1231/1827 (67.4%) | 1275/1827 (69.8%) | 1323/1827 (72.4%) | +0.4 pts | +2.4 pts | +2.6 pts |
| Number selection questions | 505/640 (78.9%) | 514/640 (80.3%) | 522/640 (81.6%) | 537/640 (83.9%) | +1.4 pts | +1.3 pts | +2.3 pts |
| Program questions | 718/1187 (60.5%) | 717/1187 (60.4%) | 753/1187 (63.4%) | 786/1187 (66.2%) | -0.1 pts | +3.0 pts | +2.8 pts |
| Simple conversations | 782/1163 (67.2%) | 798/1163 (68.6%) | 819/1163 (70.4%) | 862/1163 (74.1%) | +1.4 pts | +1.8 pts | +3.7 pts |
| Hybrid conversations | 441/664 (66.4%) | 433/664 (65.2%) | 456/664 (68.7%) | 461/664 (69.4%) | -1.2 pts | +3.5 pts | +0.8 pts |
| Hybrid first part | 258/363 (71.1%) | 248/363 (68.3%) | 271/363 (74.7%) | 280/363 (77.1%) | -2.8 pts | +6.3 pts | +2.5 pts |
| Hybrid second part | 183/301 (60.8%) | 185/301 (61.5%) | 185/301 (61.5%) | 181/301 (60.1%) | +0.7 pts | +0.0 pts | -1.3 pts |
| Turn 0 | 385/500 (77.0%) | 389/500 (77.8%) | 392/500 (78.4%) | 403/500 (80.6%) | +0.8 pts | +0.6 pts | +2.2 pts |
| Turn 1 | 349/500 (69.8%) | 348/500 (69.6%) | 358/500 (71.6%) | 377/500 (75.4%) | -0.2 pts | +2.0 pts | +3.8 pts |
| Turn 2 | 238/376 (63.3%) | 240/376 (63.8%) | 250/376 (66.5%) | 261/376 (69.4%) | +0.5 pts | +2.7 pts | +2.9 pts |
| Turn 3 | 153/266 (57.5%) | 146/266 (54.9%) | 165/266 (62.0%) | 174/266 (65.4%) | -2.6 pts | +7.1 pts | +3.4 pts |
| Turn 4 | 67/132 (50.8%) | 70/132 (53.0%) | 76/132 (57.6%) | 73/132 (55.3%) | +2.3 pts | +4.5 pts | -2.3 pts |

Story from the table:

```text
v1 -> v2 showed that evidence selection alone is useful but weak.
v2 -> v3 showed that verification/retry is a stronger improvement.
v3 -> v4 showed that retrieval-guided reasoning examples can add another meaningful gain, but with tradeoffs.
```

Interpretation:

| Observation | Meaning |
| --- | --- |
| v2 only improves `+0.4 pts` over v1. | Evidence reranking helps some cases, but the cost/latency is hard to justify by itself. |
| v3 improves `+2.4 pts` over v2. | No-gold verification retry is a strong, practical improvement. This supports the failure analysis around dirty answers, zero selection, and suspicious calculations. |
| v4 improves `+2.6 pts` over v3. | Similar train examples help on this train-500 setting after leakage control. The v4 hypothesis is empirically promising. |
| v4 improves number selection by `+2.3 pts` and program questions by `+2.8 pts` over v3. | Retrieval examples seem to help both lookup/value-selection and reasoning/calculation patterns. |
| v4 improves simple conversations by `+3.7 pts` over v3. | Examples help when the conversation structure is less complex. |
| v4 barely improves hybrid overall by `+0.8 pts` and regresses hybrid second part by `-1.3 pts`. | For harder multi-part conversational dependencies, retrieved examples may add noise or fail to resolve the real context dependency. |
| v4 improves Turns 0-3, but regresses Turn 4. | It helps earlier/mid turns but is less reliable deeper in the conversation. |

Honest conclusion:

```text
v3 is the cleanest robust improvement.
v4 is promising and empirically better on both train-500 and held-out dev, but it is more complex.
The version progression shows disciplined iteration:
1. establish baseline,
2. improve grounding,
3. add no-gold verification,
4. test retrieval-guided reasoning as a controlled extension.
```

### Dev Validation Results

Run setting:

```text
split: dev
records: 421
turns: 1490
model: gpt-4o-mini
metric: strict execution accuracy against executed_answers
```

| Breakdown | v1 | v2 | v3 | v4 | v2-v1 | v3-v2 | v4-v3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full results | 987/1490 (66.2%) | 1022/1490 (68.6%) | 1052/1490 (70.6%) | 1070/1490 (71.8%) | +2.4 pts | +2.0 pts | +1.2 pts |
| Number selection questions | 377/487 (77.4%) | 395/487 (81.1%) | 400/487 (82.1%) | 394/487 (80.9%) | +3.7 pts | +1.0 pts | -1.2 pts |
| Program questions | 610/1003 (60.8%) | 627/1003 (62.5%) | 652/1003 (65.0%) | 676/1003 (67.4%) | +1.7 pts | +2.5 pts | +2.4 pts |
| Simple conversations | 724/1052 (68.8%) | 754/1052 (71.7%) | 768/1052 (73.0%) | 787/1052 (74.8%) | +2.9 pts | +1.3 pts | +1.8 pts |
| Hybrid conversations | 263/438 (60.0%) | 268/438 (61.2%) | 284/438 (64.8%) | 283/438 (64.6%) | +1.2 pts | +3.6 pts | -0.2 pts |
| Hybrid conversations first part | 154/250 (61.6%) | 154/250 (61.6%) | 162/250 (64.8%) | 159/250 (63.6%) | +0.0 pts | +3.2 pts | -1.2 pts |
| Hybrid conversations second part | 109/188 (58.0%) | 114/188 (60.6%) | 122/188 (64.9%) | 124/188 (66.0%) | +2.6 pts | +4.3 pts | +1.1 pts |
| Turn 0 | 306/421 (72.7%) | 314/421 (74.6%) | 314/421 (74.6%) | 322/421 (76.5%) | +1.9 pts | +0.0 pts | +1.9 pts |
| Turn 1 | 294/421 (69.8%) | 300/421 (71.3%) | 309/421 (73.4%) | 313/421 (74.3%) | +1.5 pts | +2.1 pts | +0.9 pts |
| Turn 2 | 192/305 (63.0%) | 196/305 (64.3%) | 206/305 (67.5%) | 208/305 (68.2%) | +1.3 pts | +3.2 pts | +0.7 pts |
| Turn 3 | 127/211 (60.2%) | 138/211 (65.4%) | 143/211 (67.8%) | 149/211 (70.6%) | +5.2 pts | +2.4 pts | +2.8 pts |
| Turn 4 | 56/108 (51.9%) | 63/108 (58.3%) | 65/108 (60.2%) | 63/108 (58.3%) | +6.4 pts | +1.9 pts | -1.9 pts |
| Turn 5 | 11/20 (55.0%) | 10/20 (50.0%) | 13/20 (65.0%) | 13/20 (65.0%) | -5.0 pts | +15.0 pts | +0.0 pts |
| Turn 6 | 1/3 (33.3%) | 1/3 (33.3%) | 2/3 (66.7%) | 2/3 (66.7%) | +0.0 pts | +33.4 pts | +0.0 pts |
| Turn 7 | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) | +0.0 pts | +0.0 pts | +0.0 pts |

Main reading:

```text
v1 -> v2 gives a solid +2.4 point dev gain, stronger than the train-500 gain.
v2 -> v3 adds another +2.0 points, mainly improving program and hybrid questions.
v3 -> v4 adds +1.2 points overall, mostly from program questions and simple conversations.
v4 is best overall on dev: 1070/1490 = 71.8%.
```

Nuance:

```text
v4 regresses number-selection questions slightly versus v3, but improves program questions enough to win overall.
The dev result validates the train-500 direction: every version step improves headline accuracy, and v4 remains the best version.
```

### V5 Dev Update

After completing the v5 dev run:

```text
split: dev
records: 421
turns: 1490
model: gpt-4o-mini
metric: strict execution accuracy against executed_answers
v5 eval path: outputs/dev_v5/eval_dev_gpt4o_mini_v5_strict_executed.jsonl
```

| Breakdown | v1 | v2 | v3 | v4 | v5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full results | 987/1490 (66.2%) | 1022/1490 (68.6%) | 1052/1490 (70.6%) | 1070/1490 (71.8%) | 1066/1490 (71.5%) |
| Number selection | 377/487 (77.4%) | 395/487 (81.1%) | 400/487 (82.1%) | 394/487 (80.9%) | 378/487 (77.6%) |
| Program | 610/1003 (60.8%) | 627/1003 (62.5%) | 652/1003 (65.0%) | 676/1003 (67.4%) | 688/1003 (68.6%) |
| Simple | 724/1052 (68.8%) | 754/1052 (71.7%) | 768/1052 (73.0%) | 787/1052 (74.8%) | 779/1052 (74.0%) |
| Hybrid | 263/438 (60.0%) | 268/438 (61.2%) | 284/438 (64.8%) | 283/438 (64.6%) | 287/438 (65.5%) |
| Hybrid first | 154/250 (61.6%) | 154/250 (61.6%) | 162/250 (64.8%) | 159/250 (63.6%) | 154/250 (61.6%) |
| Hybrid second | 109/188 (58.0%) | 114/188 (60.6%) | 122/188 (64.9%) | 124/188 (66.0%) | 133/188 (70.7%) |

Turn-level:

| Turn | v1 | v2 | v3 | v4 | v5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turn 0 | 306/421 (72.7%) | 314/421 (74.6%) | 314/421 (74.6%) | 322/421 (76.5%) | 312/421 (74.1%) |
| Turn 1 | 294/421 (69.8%) | 300/421 (71.3%) | 309/421 (73.4%) | 313/421 (74.3%) | 313/421 (74.3%) |
| Turn 2 | 192/305 (63.0%) | 196/305 (64.3%) | 206/305 (67.5%) | 208/305 (68.2%) | 208/305 (68.2%) |
| Turn 3 | 127/211 (60.2%) | 138/211 (65.4%) | 143/211 (67.8%) | 149/211 (70.6%) | 151/211 (71.6%) |
| Turn 4 | 56/108 (51.9%) | 63/108 (58.3%) | 65/108 (60.2%) | 63/108 (58.3%) | 66/108 (61.1%) |
| Turn 5 | 11/20 (55.0%) | 10/20 (50.0%) | 13/20 (65.0%) | 13/20 (65.0%) | 14/20 (70.0%) |
| Turn 6 | 1/3 (33.3%) | 1/3 (33.3%) | 2/3 (66.7%) | 2/3 (66.7%) | 2/3 (66.7%) |
| Turn 7 | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) | 0/1 (0.0%) |

Main reading:

```text
v4 remains the best overall dev score: 1070/1490 = 71.8%.
v5 is very close overall: 1066/1490 = 71.5%, only 4 turns behind v4.
v5 improves program questions over v4: 688/1003 = 68.6% versus 676/1003 = 67.4%.
v5 also improves hybrid conversations and later-turn performance, especially hybrid second-part questions.
v5 regresses number-selection questions, simple conversations, and Turn 0 compared with v4.
```

Interpretation:

```text
v4 is still the best pure accuracy version on dev.
v5 is the better engineering/auditability version: it executes structured calculation plans, exposes values and evidence ids, and reduces dependence on free-text arithmetic.
For reporting, v5 should be framed as a structured-execution improvement rather than a headline accuracy win.
```

### Solving V1's Original Errors

This analysis uses v1's original wrong turns as the denominator.

```text
v1 total errors: 604
v1 correct turns: 1223
total turns: 1827
```

| Version | v1 errors solved | % of v1 errors solved | v1 errors still wrong | % still wrong |
| --- | ---: | ---: | ---: | ---: |
| v2 | 113/604 | 18.7% | 491/604 | 81.3% |
| v3 | 135/604 | 22.4% | 469/604 | 77.6% |

Regression check against v1's originally correct turns:

| Version | v1 correct cases regressed | % of v1 correct regressed | v1 correct kept correct |
| --- | ---: | ---: | ---: |
| v2 | 105/1223 | 8.6% | 1118/1223 |
| v3 | 83/1223 | 6.8% | 1140/1223 |

Relationship between v2 and v3 fixes:

```text
Both v2 and v3 solved: 78
Only v2 solved: 35
Only v3 solved: 57
Neither solved: 434
```

Main reading:

```text
v3 solves more of v1's original errors than v2.
v3 also causes fewer regressions from v1's originally correct cases.
This suggests v3 is a cleaner improvement path than v2 alone.
```

### Industry-Style Significance Interpretation

| Version comparison | Overall gain | Interpretation |
| --- | ---: | --- |
| v2 vs v1 | +0.4 points | Small / marginal. Useful evidence that evidence selection can help, but not strong enough alone to justify the added latency/cost. |
| v3 vs v2 | +2.4 points | Meaningful. This is a real gain, especially because it improves harder program questions. |
| v3 vs v1 | +2.8 points | Solid prototype improvement. Worth reporting, but not a production-level breakthrough. |

Important nuance:

```text
v2 adds an extra LLM call for evidence reranking, but improves only from 66.9% to 67.4%.
In an industry setting, that would usually be questioned unless it improves a high-value slice or reduces severe errors.
```

v3 is better justified:

```text
v1: 1223/1827 = 66.9%
v3: 1275/1827 = 69.8%
gain: +52 correct turns
```

v3 also has fewer regressions than v2:

```text
v2 regressed 105 originally-correct v1 turns.
v3 regressed 83 originally-correct v1 turns.
```

Overall interpretation:

```text
v2 is a useful stepping-stone, but not independently impressive.
v3 is a meaningful improvement and worth presenting as the final implemented method.
For a real industry decision, we would still want repeated runs, a confidence interval, dev-set validation, or a paired significance test.
For this assignment, v3's +2.8 points over v1 is a credible improvement story.
```

## V4 Implementation

Implemented v4 as:

```text
v4 = v3 + lightweight train-example retrieval
```

What changed:

| Component | v4 behavior |
| --- | --- |
| Current-record evidence | Same as v2/v3: record-local evidence snippets plus LLM reranking. |
| Verification retry | Same as v3: one no-gold retry when the draft answer looks suspicious. |
| Train-example retrieval | New: local lexical retrieval over solved train turns, with no vector DB and no extra API call. |
| Prompt | Adds a compact "Similar solved train examples" block only for v4. |
| Inspection | `chat --show-examples` prints the retrieved train examples used by v4. |

Guardrails:

```text
Retrieve only from train examples.
For dev evaluation, retrieving from train is allowed.
For train-sample evaluation, exclude every selected evaluation record/page from the retrieval pool.
Also exclude the current record_id and same underlying PDF page at retrieval time.
Use retrieved examples only as reasoning-pattern hints.
Do not copy numbers from retrieved examples.
The current answer must still come from the selected record and selected evidence.
```

Implementation files:

```text
src/example_retrieval.py: builds the lexical train-example index and retrieves examples, excluding the current record/page.
src/answers.py: adds AnswerVersion.V4, injects retrieved examples before answering, and returns them for inspection.
src/prompts.py: formats examples into the prompt only when v4 provides them.
src/main.py: exposes --version v4 for chat/run, --show-examples for chat inspection, and removes selected train-evaluation pages from the v4 retrieval pool.
tests/test_example_retrieval.py: covers retrieval, same-record/page exclusion, and formatting guardrails.
tests/test_evaluation.py: covers the train-sample no-leakage retrieval pool and dev-from-train retrieval pool.
```

## V5 Implementation: Structured Calculation Execution

Implemented v5 as:

```text
v5 = evidence selection + no-gold verification retry + auditable structured calculation execution
```

This means v5 keeps:

```text
record-local evidence selection
+ no-gold verification retry
+ deterministic execution of a model-proposed calculation plan
```

The design intentionally builds v5 on v3 rather than v4. V4's retrieved train examples are useful, but they add prompt length, retrieval-pool guardrails, and possible example noise. V5 is a cleaner test of whether deterministic arithmetic helps after evidence selection and verification.

Formula:

Instead of trusting free-text arithmetic, v5 asks the model to output a small auditable JSON calculation plan. The important change from the first v5 draft is that source values are named and can point back to selected evidence:

```json
{
  "values": [
    {"id": "current_revenue", "value": 206588, "evidence": "E1"},
    {"id": "prior_revenue", "value": 181001, "evidence": "E2"}
  ],
  "steps": [
    {"id": "change", "op": "subtract", "args": ["current_revenue", "prior_revenue"]},
    {"id": "change_rate", "op": "divide", "args": ["change", "prior_revenue"]}
  ],
  "answer": "change_rate"
}
```

The local executor computes:

```text
current_revenue = 206588 from E1
prior_revenue = 181001 from E2
change = current_revenue - prior_revenue = 25587
change_rate = change / prior_revenue = 0.14136
answer = change_rate
Final answer: 0.14136
```

This makes v5 less paper-like than the raw operation-only draft. The paper-style program view is mostly:

```text
divide(subtract(206588, 181001), 181001)
```

The revised v5 view is:

```text
extract named evidence-grounded financial variables -> execute a small calculation graph
```

The executor still supports the old draft schema for backwards compatibility:

```json
{
  "steps": [
    {"id": 0, "op": "subtract", "args": [206588, 181001]},
    {"id": 1, "op": "divide", "args": ["#0", 181001]}
  ],
  "answer": "#1"
}
```

Supported operations:

| Operation | Formula |
| --- | --- |
| `select(x)` | `x` |
| `add(a, b)` | `a + b` |
| `subtract(a, b)` | `a - b` |
| `multiply(a, b)` | `a * b` |
| `divide(a, b)` | `a / b` |
| `negate(x)` | `-x` |
| `abs(x)` | `|x|` |
| `max(...)` | maximum argument |
| `min(...)` | minimum argument |

Step references can use semantic ids such as `"change"` and `"change_rate"`. Old numeric step references such as `"#0"` and `"#1"` are still accepted.

Common ConvFinQA formulas:

```text
difference = subtract(current, previous)
percentage change = divide(subtract(current, previous), previous)
part of total = divide(part, total)
maximum allowed amount = multiply(base, rate)
```

For percentage, portion, ratio, and rate questions, v5 prefers executable ratio scale:

```text
4.7 / 150 = 0.03133
```

not display percent form:

```text
3.133%
```

Implementation files:

```text
src/prompts.py: when use_structured_calculation=True, asks for an auditable Calculation plan JSON block with named values and optional evidence ids.
src/calculation_plan.py: validates and executes the JSON plan with a closed operation set, named source values, semantic step ids, and old numeric step references.
src/offline_fallback.py: provides a limited offline numeric fallback that generates simple v5-shaped candidates from selected evidence, scores them, and returns an answer only when confidence is high enough.
src/answers.py: wires AnswerVersion.V5 for clean structured execution and AnswerVersion.V5A for the limited offline fallback variant.
src/main.py: exposes --version v5 and --version v5a in chat/run.
tests/test_calculation_plan.py: covers named values, evidence ids, step references, final-answer replacement, duplicate ids, and invalid-plan fallback.
tests/test_offline_fallback.py: covers the no-LLM fallback path for simple selection, capacity multiplication, and low-confidence abstention.
tests/test_answer_versions.py: confirms v5 stays clean while only v5a uses offline fallback.
tests/test_evaluation.py: confirms v5/v5a do not use the v4 train-example retrieval pool.
```

Answer flow:

```text
1. Select record-local evidence, same as v2/v3.
2. Ask the model for Target, Values, Operation, Calculation plan JSON, Final answer, and Calculation.
3. Run the v3 no-gold verifier/retry if the draft looks suspicious.
4. Extract and validate the calculation-plan JSON.
5. Execute the plan locally.
6. Replace the model's Final answer line with the deterministic result.
```

## V5A Implementation: Limited Offline Fallback

Implemented v5a as:

```text
v5a = v5 + limited offline numeric fallback
```

This keeps v5 clean for the main structured-execution experiment and uses v5a for robustness/confidence exploration.

Offline fallback flow:

```text
1. Trigger only for v5a when the OpenAI call fails or no executable v5a plan is found.
2. If evidence reranking cannot call the API, use lexical candidate evidence selection.
3. Extract numeric values and nearby labels from selected evidence.
4. Generate simple candidate plans: select, divide, subtract, and multiply-by-rate.
5. Execute candidates with the same calculation-plan executor.
6. Score candidates using evidence rank, label/question overlap, operation cues, date penalties, zero-value penalties, and domain-specific cues.
7. Return the top candidate only if confidence is high enough; otherwise abstain with `unable to determine with offline fallback`.
```

This covers a limited modeling fallback strategy:

```text
The system can still answer simple numeric selection/arithmetic cases when the LLM is unavailable, but it does not claim to solve complex ConvFinQA reasoning offline.
```

This is not full ConvFinQA DSL generation. The model still chooses the relevant values and operation, but code performs the final arithmetic. The goal is to reduce errors where the model found the right numbers but produced the wrong final calculation or wrong percentage scale.

Why v5 may help:

| Error pattern | V5 behavior |
| --- | --- |
| Right values, arithmetic slip | Local execution computes the plan exactly. |
| Percent display instead of ratio scale | Prompt asks for ratio-scale plans such as `divide(part, total)`. |
| Final answer contradicts stated calculation | The executed plan replaces the final answer. |
| Rounding / formatting drift | `_format_number` writes a compact deterministic numeric value. |

Limitations:

| Limitation | Meaning |
| --- | --- |
| Wrong values | The executor will faithfully compute the wrong selected values. |
| Wrong operation | The executor will faithfully compute the wrong operation. |
| Missing or invalid JSON plan | V5 falls back to the original model answer. |
| Non-numeric answers | The structured executor is only useful for numeric cases. |

Review note before large runs:

```text
The current extractor executes the first valid calculation-plan-shaped JSON object in the answer.
Before expensive v5 runs, it should be hardened to prefer JSON following the explicit "Calculation plan JSON:" label.
It may also need a policy decision on whether {"steps":[],"answer":0} is acceptable for number-selection cases.
```

Future extension: if v5 beats v3 cleanly, a later version can test combining v5 with v4-style train-example retrieval.

## Future V6: Structured Conversation State

A natural v6 direction is:

```text
v6 = v5 + structured conversation-state memory
```

This is easier to add after v5 because v5 already introduces a small DAG-like calculation plan instead of only free-text answers. For example:

```json
{
  "values": [
    {"id": "drawn_amount", "value": 4.7, "evidence": "T-33"},
    {"id": "credit_facility_amount", "value": 150, "evidence": "T-31"}
  ],
  "steps": [
    {"id": "ratio", "op": "divide", "args": ["drawn_amount", "credit_facility_amount"]}
  ],
  "answer": "ratio"
}
```

This is effectively:

```text
drawn_amount ┐
              ├─ divide -> ratio
facility_amount ┘
```

Because the plan exposes named values, evidence ids, and dependencies, v6 could persist useful nodes as structured conversation state:

```json
{
  "turn": 2,
  "variables": {
    "drawn_amount": 4.7,
    "credit_facility_amount": 150,
    "ratio": 0.03133
  },
  "evidence": {
    "drawn_amount": "T-33",
    "credit_facility_amount": "T-31"
  }
}
```

This would help later questions such as:

```text
what percentage, then, did that amount represent?
```

Instead of relying only on compact text history like `Final answer: 4.7`, v6 could resolve:

```text
that amount = drawn_amount
denominator = credit_facility_amount
operation = drawn_amount / credit_facility_amount
```

So the extension path is:

```text
v5: build and execute a calculation DAG for one turn.
v6: persist useful DAG nodes as structured state across turns.
```

This targets one of the largest remaining unsolved modeling gaps after grounding, verification, and deterministic execution: ambiguous multi-turn state tracking.

## Debuggability And Verification

The implementation is intentionally easier to debug than a heavier agent-style system.

| Design choice | Why it helps |
| --- | --- |
| Versioned pipeline | Compare `v1`, `v3`, `v4`, and `v5` to see which change caused what. |
| Raw run files | Predictions are saved before scoring, so model outputs can be inspected directly. |
| Separate evaluation | Parser/scoring changes can be tested without rerunning model calls. |
| Selected evidence display | `--show-evidence` shows whether the model saw the right facts. |
| v4 example display | `--show-examples` shows which train reasoning examples were retrieved. |
| v5 JSON plans | Values, evidence ids, operations, and executed answers are inspectable. |
| v5a fallback confidence | When fallback triggers, it reports confidence and candidate-derived output. |
| Unit tests | Tests cover parsing, evidence, verification, calculation plans, fallback, and evaluation. |
| Thin CLI | `main.py` delegates behavior to focused modules. |
| No hidden agent loop | There is no opaque multi-agent planning loop changing behavior unpredictably. |

Typical debug path:

```text
1. Did evidence selection find the right snippets?
2. Did the prompt produce the right values?
3. Did verification retry when it should?
4. Did v5 produce valid JSON?
5. Did the executor compute the right value?
6. Did evaluation parse the final answer correctly?
```

The most debuggable path is v5 because each answer can expose:

```text
values
evidence ids
operation
calculation plan JSON
final answer
```

When v5 fails, the failure can usually be localized to one of:

```text
wrong evidence
wrong value extraction
wrong operation
bad JSON
executor issue
parser issue
```

This is a useful assignment signal: the solution is not just more complex, it is inspectable and verifiable.

## Development Evidence Trail

This section records the observed examples/statistics that triggered each
version evolution, plus the within-version tweaks made during development.

### Version Evolution Drivers

| Step | Observed example / statistic | What it told us | Decision |
| --- | --- | --- | --- |
| v1 baseline | v1 scored `1223/1827 (66.9%)` on 500 random42 train records. | Full-record prompting works reasonably, but leaves many errors. | Keep v1 as the baseline. |
| v1 error analysis | Early error counts: calculation/program `309`, ratio/percentage `231`, number-selection `110`, refusal/non-numeric `65`. | Errors were not only missing context; there were also wrong operations, formatting, and arithmetic issues. | Add focused improvements instead of jumping straight to heavy program generation. |
| v1 -> v2 | `Double_C/2008/page_217.pdf`, Turn 0: v1 used wrong partial values and answered about `0.26`; gold was `0.30895`. | The model sometimes needed focused evidence and the right numerator/denominator. | Add record-local evidence selection. |
| v2 result | v2 scored `1231/1827 (67.4%)`, only `+0.4 pts` over v1. | Evidence helped, but not enough by itself. | Treat v2 as useful but not final. |
| v2 regression example | `Double_ETR/2016/page_424.pdf`, Turn 2: v1 got `3.13%` correct, while v2 followed the wrong `cash borrowings = 0` interpretation and answered `0`. | Evidence can reinforce the wrong interpretation. Need verification, not just retrieval. | Build v3 verification retry. |
| v2 -> v3 | `Double_ETR`, Turn 0: v1/v2 chose `0`; the same evidence sentence also had `$4.7 million of letters of credit outstanding`. | Need a no-gold rule to catch suspicious zero selection. | Add v3 no-gold verifier/retry. |
| v3 result | v3 scored `1275/1827 (69.8%)`, `+2.4 pts` over v2 and `+2.8 pts` over v1. | Verification retry gives a meaningful gain. | Treat v3 as the strongest current implemented version. |
| v3 remaining errors | v3 had `552` wrong turns; program-question wrong turns were `434/552 (78.6%)`. | Remaining errors are mostly reasoning/program errors, not simple lookup. | Consider future retrieval examples or stronger checking. |
| v3 -> v4 motivation | Rough v4-target reasoning-pattern candidates: `226/552 (40.9%)` of v3 wrong turns. | Similar examples may help operation patterns, but not guaranteed. | Implement v4 as an experiment: v3 + train-example retrieval. |

### Within-Version Tuning Drivers

| Area | Observed issue | Example / statistic | Tweak made |
| --- | --- | --- | --- |
| Evaluation standard | False negatives from percent display. | `Double_C/2008/page_217.pdf`, Turn 3: `5.3%` / `5.4%` should match executed `0.05346`. | Parser normalizes `%` to ratio scale and supports tolerance. |
| Evaluation standard | `conv_answers` could conflict with paper-style scoring. | We decided strict gold should be `executed_answers`, not display `conv_answers`. | Evaluation uses strict executed-answer gold. |
| Batch running | API rate limit caused runs to stop and risk losing work. | 429 RPD/TPM errors during 500-case runs. | Added checkpoint append per completed turn, selected-record sidecar, progress display, and `--resume`. |
| Parallel running | Needed faster 500-case runs. | 4 workers was faster but increased rate-limit risk. | Added `--workers`; kept evaluation compatible and resumable. |
| v2 evidence | Evidence selection helped a true wrong-evidence case. | `Double_C`, Turn 0: v1 wrong `0.26`; v2 correct `0.309`. | Kept evidence selection as v2/v3/v4 base. |
| v2 evidence | Evidence can still mislead or fail. | `Double_ETR`: v2 still chose `0` and propagated it. | Did not rely on evidence alone; added verifier in v3. |
| v3 verifier | Dirty final answer caused wrong parse. | `Double_ETR`, Turn 1: model mentioned `$150 million`, but parser grabbed another number/date. | Added clean `Final answer:` retry rule. |
| v3 verifier | Zero selection trap. | `cash borrowings = 0` vs `letters of credit outstanding = 4.7`. | Added generic zero-candidate retry rule. |
| v3 verifier | Model sometimes omitted the competing value from `Values:`. | v4 run showed `Values: cash borrowings = 0` only, while selected evidence had `4.7`. | Made zero-selection verifier inspect selected evidence too. |
| v3 verifier | Percent rounding caused unnecessary retry. | `3.1%` vs calculation `3.1333%`. | Relaxed percent-specific calculation tolerance. |
| v4 retrieval | Retrieved examples were invisible. | Could not tell which examples v4 used. | Added `--show-examples`. |
| v4 retrieval | Train-sample leakage risk. | Random42 train evaluation could retrieve examples from the same 500 selected records. | Exclude selected train evaluation records/pages from v4 retrieval pool. |
| v4 retrieval | `Single` / `Double` same-page leakage. | `Double_X/page.pdf` could retrieve `Single_X/page.pdf-*`. | Added same underlying PDF page exclusion. |
| v4 retrieval | Multiple examples can conflict. | Top 3 examples may imply different operations. | Added prompt rule: examples are ranked; if they conflict, follow the higher-ranked example. |
| v4 retrieval | Full vector RAG may be overkill. | Dataset size: median formatted record `705` words, p99 `1551`, max `2362`. | Kept retrieval lightweight; no vector DB. |
| Conversation history | Full `Target/Values/Operation/Final answer/Calculation` blocks are noisy when fed back into later turns. | Feedback noted that verbose reasoning history can confuse follow-up references. | Store only compact `Final answer: <value>` in conversation history while preserving raw predictions in output JSONL. |

### Most Important Concrete Examples

| Example | What happened | What it changed |
| --- | --- | --- |
| `Double_C/2008/page_217.pdf`, Turn 0 | v1 was wrong; v2/v3 were correct. | Justified evidence selection. |
| `Double_ETR/2016/page_424.pdf`, Turn 0 | v1/v2 answered `0`; v3/v4 answered `4.7`. | Justified zero-selection verification. |
| `Double_ETR`, Turn 1 | v1/v2 mentioned `150` but parsed wrong; v3/v4 produced clean `150`. | Justified clean final-answer retry. |
| `Double_ETR`, Turn 2 | v2 propagated `0`; v3/v4 used `4.7 / 150 = 3.1%`. | Showed verifier can prevent propagation. |
| `Single_ABMD/2006/page_62.pdf-1` | v1/v2/v3 still failed direction/reference turns. | Shows current verifier is not enough; motivates future checker/reasoning work. |
| `Double_ADBE/2011/page_83.pdf` | Gold executable answer was `no`, while model answered numeric `10`. | Dataset/task-format ambiguity; do not overfit. |

Overall development logic:

```text
v1 showed baseline capability and failure types.
v2 targeted wrong evidence / number selection.
v3 targeted suspicious answer reasoning and formatting.
v4 explored retrieval-guided reasoning patterns, with leakage controls.
Future v5 should likely be v3 + stronger deterministic checker, because v4 is still experimental.
```

## Current Coverage Of Hard Error Types

| Problem | v1 | v2 | v3 | v4 |
| --- | --- | --- | --- | --- |
| Wrong evidence | No | Partly | Partly | Partly |
| Wrong value selection | No | Partly | Better than v2 | Better overall on dev, though some lookup cases regress. |
| Wrong operation direction | No | No | Slightly / limited | Helps some program-question patterns, but not a full solution. |
| Ambiguous follow-up references | Basic history only | Basic history + evidence | Better if verifier catches propagation | Helps overall accuracy, but deeper hybrid turns remain hard. |
| Dataset annotation mismatch | No | No | No | No |

More detail:

| Problem | Current best handling | Example |
| --- | --- | --- |
| Wrong evidence | v2/v3 evidence selection can help when it surfaces the right row/snippet. | `Double_C/2008/page_217.pdf`, Turn 0: v1 wrong, v2/v3 correct. |
| Wrong value selection | v3 helps when the wrong value is locally suspicious, such as zero vs non-zero candidate. | `Double_ETR/2016/page_424.pdf`, Turn 0: v1/v2 `0`, v3/v4 `4.7`. |
| Wrong operation direction | Mostly unsolved. v3 can catch contradiction between stated operation and final answer, but not if the operation itself is wrong. | `Single_ABMD/2006/page_62.pdf-1`: v1/v2/v3 still wrong. |
| Ambiguous follow-up references | Partly handled by conversation history and compact final-answer history. v3 helps if bad reference leads to suspicious output. | `Double_ETR`, Turn 2 improved after Turn 0 was fixed. |
| Dataset annotation mismatch | Not solved; should be documented as a limitation rather than overfit. | `Double_ADBE/2011/page_83.pdf`, gold `no`, model answers `10`. |

Summary:

```text
v2/v3 improve evidence and value selection.
v3 improves some follow-up propagation issues.
v4 helps overall on dev, especially program questions, but it still does not fully solve operation direction or deeper conversational dependencies.
Wrong operation direction and dataset annotation mismatch remain largely unsolved.
```

## V1 / V2 / V3 Representative Error Examples

These are the same representative examples used during failure analysis.

| Example | Turn | Gold | v1 | v2 | v3 | Result |
| --- | ---: | ---: | --- | --- | --- | --- |
| `Single_ABMD/2006/page_62.pdf-1` calculation direction | 2 | `78` | `-78`, wrong | `-78`, wrong | `-64`, wrong | v3 still does not solve ambiguous direction / follow-up reference. |
| `Single_ABMD/2006/page_62.pdf-1` propagated calculation | 3 | `78000` | `-78000`, wrong | `-78000`, wrong | `167000`, wrong | v3 changed the error, but still wrong. |
| `Double_C/2008/page_217.pdf` wrong evidence / ratio | 0 | `0.30895` | `0.26`, wrong | `0.309`, correct | `0.309`, correct | v2 fixed this; v3 preserved it. |
| `Double_C/2008/page_217.pdf` percent formatting | 3 | `0.05346` | `0.054`, correct | `0.053`, correct | `0.053`, correct | All correct under latest evaluator. |
| `Double_ETR/2016/page_424.pdf` number selection | 0 | `4.7` | `0`, wrong | `0`, wrong | `4.7`, correct | v3 fixed the zero-selection issue. |
| `Double_ETR/2016/page_424.pdf` credit facility value | 1 | `150` | parsed `0.5`, wrong | parsed `2021`, wrong | `150`, correct | v3 fixed dirty / non-clean final answer. |
| `Double_ETR/2016/page_424.pdf` follow-up percentage | 2 | `0.03133` | `0.0313`, correct | `0`, wrong | `0.0313`, correct | v3 recovered because turn 0 was fixed. |
| `Double_ETR/2016/page_424.pdf` max letters of credit | 3 | `75` | `75`, correct | `75`, correct | `75`, correct | All correct. |
| `Double_ADBE/2011/page_83.pdf` annotation mismatch | 2 | `no` | `10`, wrong | `10`, wrong | `10`, wrong | Still wrong; this is dataset/task-format ambiguity. |
| `Double_ADBE/2011/page_83.pdf` amortization rate | 3 | `16.66667` | refusal, wrong | refusal, wrong | refusal, wrong | v3 did not fix this case. |

Main reading:

```text
v3 fixes the Double_ETR failures that motivated the no-gold verification retry.
v3 keeps the Double_C evidence-selection win from v2.
v3 does not solve every calculation/reference ambiguity, especially Single_ABMD.
v3 also does not solve annotation-style mismatch cases like Double_ADBE.
```

## Pain Points And Version Coverage

| Pain point | What goes wrong | Addressed by | How it is resolved |
| --- | --- | --- | --- |
| Full document has too many competing numbers | Model picks the wrong row, sentence, or value from the record. | `v2` | Adds record-local evidence selection to highlight relevant text sentences, table rows, and number-centered snippets before answering. |
| Follow-up questions depend on previous turns | Questions like "that amount" or "during that period" need conversation context. | `v1` | Keeps prior user questions and assistant answers in the chat history. |
| Wrong value selected from same sentence | Example: choosing `cash borrowings = 0` instead of `letters of credit outstanding = 4.7`. | `v3` | Verification detects suspicious zero-selection when non-zero alternatives are present and retries. |
| Missing or messy `Final answer:` line | Parser may grab dates, units, or unrelated numbers like `2021` or `50%`. | `v3` | Verification retries if the final answer is missing, has extra words, or contains multiple numbers. |
| Model refuses despite answer being in record | Says "not enough information" even when evidence has numeric candidates. | `v3` | No-gold verifier retries refusal-style answers when selected evidence contains answer candidates. |
| Percentage / ratio scale mismatch | Model outputs `3.1%` vs executable answer `0.031`, or mixes percent-point and ratio scale. | `v3`, improved by `v5` | `v3` checks suspicious percent formatting; `v5` asks for executable numeric plans so code can output ratio-scale answers. |
| Wrong denominator in portion questions | Model computes `total / part` instead of `part / total`. | `v3`, improved by `v5` | `v3` retries denominator-direction issues; `v5` executes a structured `divide(part, total)` plan if the model emits the right plan. |
| Arithmetic mistakes | Model finds the right values but calculates incorrectly. | `v3`, mainly `v5` | `v3` detects mismatches between operation and final answer; `v5` executes the calculation locally instead of trusting model arithmetic. |
| Operation direction errors | Example: computes `167 - 245 = -78` instead of `245 - 167 = 78`. | Partially `v3`, potentially `v5` | `v3` can catch operation/final-answer inconsistency, but if the operation itself is wrong, v5 still depends on the model choosing the right direction. |
| Reasoning-pattern uncertainty | Model struggles to infer whether a question asks for difference, ratio, percent change, etc. | `v4` | Retrieves similar solved train examples as reasoning-pattern hints, without using their numbers as evidence. |
| Later-turn error propagation | Early wrong answers affect later follow-ups. | Partially `v3` / `v4` / `v5` | Cleaner final answers, retries, examples, and deterministic execution reduce but do not fully eliminate propagation. |
| Need reproducible evaluation | Model calls are expensive and parser changes should not require rerunning everything. | Evaluation pipeline, not one version | Separates `run -> evaluate -> analyze-results`, saving raw predictions first and scoring later. |
| Need paper-style analysis | Single headline accuracy hides where the system succeeds or fails. | Evaluation pipeline | Produces breakdowns by number-selection/program questions, simple/hybrid conversations, and turn index. |

## Table Format Handling

The current solution does not build a full table parser or spreadsheet-style table engine. It treats the table content already present in each ConvFinQA record as structured text, then uses row-level evidence selection and evidence-linked value extraction.

| Stage | How tables are handled |
| --- | --- |
| Record formatting | The prompt includes the record text and table content supplied by the dataset. |
| Evidence selection | Table rows become selectable snippets with IDs such as `T-31`, so the answerer can focus on relevant rows instead of the whole record. |
| Number-centered snippets | Numeric values are surfaced with nearby labels/context, which helps when an answer depends on one table cell or row. |
| v3 verification | Suspicious outputs can trigger retry, including zero-vs-nonzero mistakes, missing final answers, messy numeric answers, and ratio/percent scale issues. |
| v5 calculation plan | Extracted values are named and linked back to evidence IDs, then arithmetic is executed in code instead of trusting free-text arithmetic. |

Summary:

```text
The solution handles tables through row-level evidence selection and evidence-linked value extraction, not through full table reconstruction.
This is a reasonable scope choice because ConvFinQA records already provide table text in a usable form.
The limitation is that the LLM still has to understand row/column alignment. If a value is ambiguous across columns, or if the correct answer requires careful joining of row labels and column headers, the system can still pick the wrong table cell.
```
