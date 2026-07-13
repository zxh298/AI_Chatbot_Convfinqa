# ConvFinQA Report

## Summary

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
