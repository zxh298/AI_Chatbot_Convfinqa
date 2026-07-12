# V1 vs V2 Comparison on Error Examples

Source files:

```text
outputs/eval_train_500_random42_gpt4o_mini_v1_strict_executed.jsonl
outputs/eval_train_500_random42_gpt4o_mini_v2_strict_executed.jsonl
```

Metric: strict execution accuracy against `executed_answers`.

These are the same representative error records used in `outputs/error_example/error_example.md`, now compared between:

```text
v1: full-record baseline
v2: record-local evidence selection
```

## 1. Calculation/Program Error

Record: `Single_ABMD/2006/page_62.pdf-1`

Why this example matters: the model selects the right values but uses the wrong subtraction direction. Evidence selection should not be expected to fix this by itself.

Turn-by-turn:

```text
Turn 0
Q: what was the balance at the beginning of the year in 2005?
Gold executed_answer: 245.0

V1 answer:
Final answer: 245
Correct: True

V2 answer:
Final answer: 245
Correct: True

Result: same, both correct.

Turn 1
Q: what was the value at the end of the year in 2006?
Gold executed_answer: 167.0

V1 answer:
Final answer: 167
Correct: True

V2 answer:
Final answer: 167
Correct: True

Result: same, both correct.

Turn 2
Q: what was the difference?
Gold executed_answer: 78.0
Gold program: subtract(245, 167)

V1 answer:
Final answer: -78
Calculation: 167 - 245 = -78
Correct: False

V2 answer:
Final answer: -78
Calculation: 167 - 245 = -78
Correct: False

Result: same, both wrong.
Why: both versions found the right numbers, but both reversed the operation direction.

Turn 3
Q: what is that value time 1000?
Gold executed_answer: 78000.0
Gold program: subtract(245, 167), multiply(#0, const_1000)

V1 answer:
Final answer: -78000
Calculation: -78 * 1000 = -78000
Correct: False

V2 answer:
Final answer: -78000
Calculation: -78 * 1000 = -78000
Correct: False

Result: same, both wrong.
Why: the sign error from turn 2 carried forward.
```

Conclusion: this is a calculation-verification problem, not an evidence-selection problem.

## 2. Ratio or Percentage Error

Record: `Double_C/2008/page_217.pdf`

Why this example matters: v2 shows a real evidence-selection win on turn 0, while the later percentage-format issue is now handled by the latest strict parser/tolerance.

Turn-by-turn:

```text
Turn 0
Q: what portion of total maximum potential amount of future payments is related to financial standby letters of credit?
Gold executed_answer: 0.30895
Gold program: divide(94.2, 304.9)

V1 answer:
Final answer: 26
Calculation used partial sub-values instead of the total 94.2.
Correct: False

V2 answer:
Final answer: 30.9
Calculation: (94.2 / 304.9) * 100 = 30.9%
Correct: True

Result: v2 fixed v1.
Why: v2 selected the correct numerator, 94.2, and denominator, 304.9.

Turn 1
Q: what about the total maximum potential amount of future payments for performance guarantees?
Gold executed_answer: 16.3

V1 answer:
Final answer: 16.3
Correct: True

V2 answer:
Final answer: 16.3
Correct: True

Result: same, both correct.

Turn 2
Q: and the total maximum potential amount of future payments?
Gold executed_answer: 304.9

V1 answer:
Final answer: 304.9
Correct: True

V2 answer:
Final answer: 304.9
Correct: True

Result: same, both correct.

Turn 3
Q: what portion us related to performance guarantees?
Gold executed_answer: 0.05346
Gold program: divide(16.3, 304.9)

V1 answer:
Final answer: 5.4
Calculation: 16.3 / 304.9 = 0.0534, expressed as percentage = 5.34%
Correct: True

V2 answer:
Final answer: 5.3
Calculation: (16.3 / 304.9) * 100 = 5.34%
Correct: True

Result: same, both correct under the latest strict evaluator.
Why: the parser now uses percentage context from the calculation line and compares against ratio-scale `executed_answers`.
```

Conclusion: v2 can fix wrong-evidence ratio cases, and the latest evaluator avoids false negatives for percentage formatting.

## 3. Number-Selection and Refusal Error

Record: `Double_ETR/2016/page_424.pdf`

Why this example matters: v2 does not fix this old failure. In one follow-up turn, v2 actually regresses because it keeps following the wrong `cash borrowings = 0` interpretation.

Turn-by-turn:

```text
Turn 0
Q: as of december 31, 2016, what was the drawn amount from the credit facility that was set to expire in august 2021?
Gold executed_answer: 4.7
Gold program: 4.7

V1 answer:
Final answer: 0
Operation: select cash borrowings
Correct: False

V2 answer:
Final answer: 0
Operation: select cash borrowings
Correct: False

Result: same, both wrong.
Why: both versions choose "no cash borrowings" instead of "$4.7 million of letters of credit outstanding."

Turn 1
Q: and what was that credit facility?
Gold executed_answer: 150.0
Gold program: 150

V1 answer:
The credit facility is in the amount of $150 million scheduled to expire in August 2021. It allows Entergy Texas to issue letters of credit against 50% of the borrowing capacity of the facility.
Parsed value: 0.5
Correct: False

V2 answer:
The credit facility was in the amount of $150 million scheduled to expire in August 2021.
Parsed value: 2021
Correct: False

Result: same outcome, both wrong.
Why: both mention the right $150 million, but neither outputs a clean machine-readable `Final answer: 150`.

Turn 2
Q: what percentage, then, did that amount represent?
Gold executed_answer: 0.03133
Gold program: divide(4.7, 150)

V1 answer:
Final answer: 3.13
Calculation: (4.7 / 150) * 100 = 3.1333...
Correct: True

V2 answer:
Final answer: 0
Calculation: 0 / 150 * 100 = 0
Correct: False

Result: v2 broke v1.
Why: v2 continued the wrong interpretation from turn 0, treating "that amount" as zero cash borrowings instead of 4.7 letters of credit.

Turn 3
Q: and under this same credit facility, what was the maximum amount of letters of credit that could be issued, in millions?
Gold executed_answer: 75.0
Gold program: multiply(150, 50%)

V1 answer:
Final answer: 75
Calculation: 150 * 0.50 = 75
Correct: True

V2 answer:
Final answer: 75
Correct: True

Result: same, both correct.
```

Conclusion: evidence selection is not automatically better. If the selected evidence or conversation state reinforces the wrong interpretation, v2 can regress.

## 4. Other / Annotation Ambiguity

Record: `Double_ADBE/2011/page_83.pdf`

Why this example matters: the model gives a reasonable human-style answer, but the gold executable answer is boolean-style. This is not a normal evidence-selection or arithmetic failure.

Turn-by-turn:

```text
Turn 0
Q: what was the weighted average useful life of purchased technology?
Gold executed_answer: 6.0

V1 answer:
Final answer: 6
Correct: True

V2 answer:
Final answer: 6
Correct: True

Result: same, both correct.

Turn 1
Q: and of customer contracts and relationships?
Gold executed_answer: 10.0

V1 answer:
Final answer: 10
Correct: True

V2 answer:
Final answer: 10
Correct: True

Result: same, both correct.

Turn 2
Q: which weighted average useful life, then, of these two was the highest?
Gold executed_answer: no
Gold program: greater(6, 10)

V1 answer:
Final answer: 10
Correct: False

V2 answer:
Final answer: 10
Correct: False

Result: same, both wrong.
Why: the natural-language question asks for the highest useful life, but the gold program encodes a yes/no comparison.

Turn 3
Q: what was the average annual amortization rate for the purchased technology segment?
Gold executed_answer: 16.66667

V1 answer:
The record does not provide enough information to determine the average annual amortization rate for the purchased technology segment.
Correct: False

V2 answer:
The record does not provide enough information regarding the average annual amortization rate for the purchased technology segment.
Correct: False

Result: same, both wrong.
```

Conclusion: this remains a residual dataset/task-format issue plus a refusal-style error. It is not solved by v2.

## Summary Table

| Error example | Key turn | Gold | V1 result | V2 result | What changed |
| --- | ---: | --- | --- | --- | --- |
| `Single_ABMD/2006/page_62.pdf-1` calculation direction | Turn 2 | `78` | `-78`, wrong | `-78`, wrong | No improvement. Both used the right numbers but reversed the subtraction. |
| `Single_ABMD/2006/page_62.pdf-1` propagated calculation | Turn 3 | `78000` | `-78000`, wrong | `-78000`, wrong | No improvement. The earlier sign error carried forward. |
| `Double_C/2008/page_217.pdf` wrong evidence/ratio | Turn 0 | `0.30895` | `26%`, wrong | `30.9%`, correct | v2 fixed this by selecting `94.2 / 304.9` instead of partial sub-values. |
| `Double_C/2008/page_217.pdf` percent formatting | Turn 3 | `0.05346` | `5.4%`, correct | `5.3%`, correct | Both are correct under latest strict parser/tolerance. |
| `Double_ETR/2016/page_424.pdf` number selection | Turn 0 | `4.7` | `0`, wrong | `0`, wrong | No improvement. Both choose "no cash borrowings" instead of letters of credit outstanding. |
| `Double_ETR/2016/page_424.pdf` credit facility value | Turn 1 | `150` | wrong parse, wrong | wrong parse, wrong | Both mention `$150 million`, but neither outputs clean `Final answer: 150`. |
| `Double_ETR/2016/page_424.pdf` follow-up percentage | Turn 2 | `0.03133` | `3.13%`, correct | `0%`, wrong | v2 regressed by continuing the wrong `cash borrowings = 0` interpretation. |
| `Double_ETR/2016/page_424.pdf` max letters of credit | Turn 3 | `75` | `75`, correct | `75`, correct | Both are correct. |
| `Double_ADBE/2011/page_83.pdf` annotation/task-format mismatch | Turn 2 | `no` | `10`, wrong | `10`, wrong | No improvement expected. Human-style answer is reasonable, but gold executable answer is boolean. |

Overall takeaway:

```text
v2 helps when the failure is genuinely wrong evidence selection.
v2 does not fix calculation-direction errors or annotation mismatches.
v2 can regress when the selected evidence reinforces a wrong interpretation from an earlier turn.
```

The next improvement should therefore be deterministic calculation and verification, not simply more evidence selection.
