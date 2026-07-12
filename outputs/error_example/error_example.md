# Error Examples From 500-Record Train Sample

Source result file: `outputs/eval_train_500_random42_full_gpt4o_mini.jsonl`

These examples illustrate the five error buckets used in the failure analysis.

## 1. Calculation/Program Error

Count: `309`

Record: `Single_ABMD/2006/page_62.pdf-1`

Original context:

```json
"table": {
  "2005": {
    "balance at the beginning of the year": 245.0,
    "accrual for warranties": 198.0,
    "warranty expense incurred for the year": -212.0,
    "balance at the end of the year": 231.0
  },
  "2006": {
    "balance at the beginning of the year": 231.0,
    "accrual for warranties": 193.0,
    "warranty expense incurred for the year": -257.0,
    "balance at the end of the year": 167.0
  }
}
```

Turn-by-turn:

```text
Turn 0
Q: what was the balance at the beginning of the year in 2005?
Gold conv_answer: 245
Gold executed_answer: 245.0
Gold program: 245
Our answer: Final answer: 245
Correct: True

Turn 1
Q: what was the value at the end of the year in 2006?
Gold conv_answer: 167
Gold executed_answer: 167.0
Gold program: 167
Our answer: Final answer: 167
Correct: True

Turn 2
Q: what was the difference?
Gold conv_answer: 78
Gold executed_answer: 78.0
Gold program: subtract(245, 167)
Our answer: Final answer: -78
Calculation: 167 - 245 = -78
Correct: False

Turn 3
Q: what is that value time 1000?
Gold conv_answer: 78000
Gold executed_answer: 78000.0
Gold program: subtract(245, 167), multiply(#0, const_1000)
Our answer: Final answer: -78000
Calculation: -78 * 1000 = -78000
Correct: False
```

Why: right numbers, wrong calculation direction.

What happened:

```text
Model selected the correct numbers -> model used the wrong calculation direction -> model generated -78 -> parser extracted -78
```

This is not a parser issue. The parser only extracted the final value that the model wrote.

## 2. Ratio or Percentage Error

Count: `231`

Record: `Double_C/2008/page_217.pdf`

Original context:

```json
"table": {
  "maximum potential amount of future payments total": {
    "financial standby letters of credit": 94.2,
    "performance guarantees": 16.3,
    "derivative instruments deemed to be guarantees": 67.9,
    "guarantees of collection of contractual cash flows": 0.3,
    "loans sold with recourse": 0.3,
    "securities lending indemnifications": 47.6,
    "credit card merchant processing": 56.7,
    "custody indemnifications and other": 21.6,
    "total": 304.9
  }
}
```

Turn-by-turn:

```text
Turn 0
Q: what portion of total maximum potential amount of future payments is related to financial standby letters of credit?
Gold conv_answer: 31%
Gold executed_answer: 0.30895
Gold program: divide(94.2, 304.9)
Our answer: Final answer: 16.1
Calculation: (49.2 / 304.9) * 100 = 16.1%
Correct: False

Turn 1
Q: what about the total maximum potential amount of future payments for performance guarantees?
Gold conv_answer: 16.3
Gold executed_answer: 16.3
Gold program: 16.3
Our answer: Final answer: 16.3
Correct: True

Turn 2
Q: and the total maximum potential amount of future payments?
Gold conv_answer: 304.9
Gold executed_answer: 304.9
Gold program: 304.9
Our answer: Final answer: 304.9
Correct: True

Turn 3
Q: what portion us related to performance guarantees?
Gold conv_answer: 5%
Gold executed_answer: 0.05346
Gold program: divide(16.3, 304.9)
Our answer: Final answer: 5.3
Calculation: (16.3 / 304.9) * 100 = 5.3%
Correct: False
```

Why: turn 3 calculation is nearly right, but final answer misses `%`, so it is parsed as `5.3` instead of `0.053`.

What happened:

```text
Model calculated a percentage -> model omitted % in the final answer -> parser treated 5.3 as a plain number -> evaluator compared 5.3 against ratio-scale gold 0.05346
```

This is why the prompt/parser/evaluator split matters: the prompt asks for clean final values, the parser defensively normalizes the text, and the evaluator compares against the executable gold scale.

## 3. Number-Selection Error

Count: `110`

Record: `Double_ETR/2016/page_424.pdf`

Original context:

```json
"doc": {
  "post_text": "see note 4 to the financial statements for a description of the money pool . entergy texas has a credit facility in the amount of $ 150 million scheduled to expire in august 2021 . the credit facility allows entergy texas to issue letters of credit against 50% ( 50 % ) of the borrowing capacity of the facility . as of december 31 , 2016 , there were no cash borrowings and $ 4.7 million of letters of credit outstanding under the credit facility . in addition , entergy texas is a party to an uncommitted letter of credit facility as a means to post collateral ."
}
```

Turn-by-turn:

```text
Turn 0
Q: as of december 31, 2016, what was the drawn amount from the credit facility that was set to expire in august 2021?
Gold conv_answer: 4.7
Gold executed_answer: 4.7
Gold program: 4.7
Our answer: Final answer: 0
Calculation: The record states there were no cash borrowings from the credit facility as of December 31, 2016.
Correct: False

Turn 1
Q: and what was that credit facility?
Gold conv_answer: 150
Gold executed_answer: 150.0
Gold program: 150
Our answer: The record does not provide enough information.
Correct: False

Turn 2
Q: what percentage, then, did that amount represent?
Gold conv_answer: 3.1%
Gold executed_answer: 0.03133
Gold program: divide(4.7, 150)
Our answer: Final answer: 0%
Calculation: The drawn amount from the credit facility was $0, which results in 0% of the total borrowing capacity.
Correct: False

Turn 3
Q: and under this same credit facility, what was the maximum amount of letters of credit that could be issued, in millions?
Gold conv_answer: 75
Gold executed_answer: 75.0
Gold program: multiply(150, 50%)
Our answer: Final answer: 75
Calculation: The credit facility allows for letters of credit against 50% of the $150 million borrowing capacity, which is $75 million.
Correct: True
```

Why: turn 0 required selecting `4.7` from the post text. The model selected `0` from “no cash borrowings” instead.

What happened:

```text
Model focused on "no cash borrowings" -> model generated 0 -> parser extracted 0
```

The `0` is not a fallback value from our code. It is the value produced by the model after selecting the wrong evidence.

## 4. Non-Numeric/Refusal Error

Count: `65`

Record: `Double_ETR/2016/page_424.pdf`

Same original context:

```json
"doc": {
  "post_text": "see note 4 to the financial statements for a description of the money pool . entergy texas has a credit facility in the amount of $ 150 million scheduled to expire in august 2021 . the credit facility allows entergy texas to issue letters of credit against 50% ( 50 % ) of the borrowing capacity of the facility . as of december 31 , 2016 , there were no cash borrowings and $ 4.7 million of letters of credit outstanding under the credit facility . in addition , entergy texas is a party to an uncommitted letter of credit facility as a means to post collateral ."
}
```

Turn-by-turn:

```text
Turn 0
Q: as of december 31, 2016, what was the drawn amount from the credit facility that was set to expire in august 2021?
Gold conv_answer: 4.7
Gold executed_answer: 4.7
Gold program: 4.7
Our answer: Final answer: 0
Correct: False

Turn 1
Q: and what was that credit facility?
Gold conv_answer: 150
Gold executed_answer: 150.0
Gold program: 150
Our answer: The record does not provide enough information.
Correct: False

Turn 2
Q: what percentage, then, did that amount represent?
Gold conv_answer: 3.1%
Gold executed_answer: 0.03133
Gold program: divide(4.7, 150)
Our answer: Final answer: 0%
Correct: False

Turn 3
Q: and under this same credit facility, what was the maximum amount of letters of credit that could be issued, in millions?
Gold conv_answer: 75
Gold executed_answer: 75.0
Gold program: multiply(150, 50%)
Our answer: Final answer: 75
Correct: True
```

Why: turn 1 is the refusal error. The answer `150` is explicitly present in the post text, but the model said there was not enough information.

What happened:

```text
Model refused even though the answer exists -> parser preserved the refusal text -> evaluator counted it as non-numeric/refusal
```

When the model does not produce a number, our parser does not replace it with `0`. It keeps the text as a non-numeric prediction.

## 5. Other

Count: `2`

Record: `Double_ADBE/2011/page_83.pdf`

Original context:

```json
"table": {
  "weighted averageuseful life ( years )": {
    "purchased technology": 6.0,
    "customer contracts and relationships": 10.0,
    "trademarks": 7.0,
    "acquired rights to use technology": 9.0,
    "localization": 1.0,
    "other intangibles": 3.0
  }
}
```

Turn-by-turn:

```text
Turn 0
Q: what was the weighted average useful life of purchased technology?
Gold conv_answer: 6
Gold executed_answer: 6.0
Gold program: 6
Our answer: Final answer: 6
Correct: True

Turn 1
Q: and of customer contracts and relationships?
Gold conv_answer: 10
Gold executed_answer: 10.0
Gold program: 10
Our answer: Final answer: 10
Correct: True

Turn 2
Q: which weighted average useful life, then, of these two was the highest?
Gold conv_answer: no
Gold executed_answer: no
Gold program: greater(6, 10)
Our answer: Final answer: 10
Calculation: The weighted average useful life of customer contracts and relationships (10 years) is higher than that of purchased technology (6 years).
Correct: False
```

Why: the gold answer is boolean/string-style `"no"`, but the model returned the larger numeric value `10`. This is not a normal numeric calculation error, so it falls into `Other`.

What happened:

```text
Model answered the value-style question it inferred -> gold expects the boolean output of greater(6, 10) -> evaluator compared 10 against "no"
```

This is a task-format mismatch. The model's explanation is semantically reasonable, but the dataset's executable answer for this comparison operation is `"no"`.

Human-reading note:

```text
Natural-language question: which weighted average useful life was highest?
Human-style answer: 10, or customer contracts and relationships
Gold program: greater(6, 10)
Gold executable answer: no
```

By human understanding, our answer `10` is reasonable. The mismatch comes from the dataset annotation: the question wording asks for the highest value, but the gold program encodes a yes/no comparison, effectively asking whether `6` is greater than `10`. This is why this example is treated as an annotation/task-format mismatch rather than a clear model reasoning failure.

## Proposed Fixes

The fixes below intentionally stay lightweight. The assignment values explainable design choices and optimal implementation within the timeframe, so the goal is to improve the main failure modes without turning the system into a full program-generation or agentic framework.

| Error type | Count | Proposed solution |
|---|---:|---|
| Calculation/program errors | 309 | Add a lightweight calculation verifier. Ask the model to output a final value and calculation, then locally re-check simple arithmetic patterns where possible. If the calculation contradicts the final answer or appears to reverse a change/difference operation, flag it or retry once. |
| Ratio or percentage errors | 231 | Strengthen final-answer normalization. If the question or calculation implies a percentage/portion/rate, require `%` in the final answer. The parser can also inspect the calculation line: if the final answer is `5.3` but the calculation says `5.3%`, normalize as a percentage. |
| Number-selection errors | 110 | Add record-local evidence selection before answering. Select the most relevant table rows and text snippets from the same record, then pass them as focused evidence. This is not full RAG; it is lightweight filtering inside the already-selected ConvFinQA record. |
| Non-numeric/refusal errors | 65 | Add an anti-refusal retry path. If the model says there is not enough information but the record contains candidate numbers, retry once with an instruction that the answer is expected to exist in the record and the model should choose the best-supported value. |
| Other | 2 | Treat as annotation or task-format ambiguity. Do not overfit special logic for rare cases where the natural-language question appears to ask for a value but the gold program expects a boolean result. |

Recommended implementation order:

1. Improve percentage parsing using the calculation line.
2. Add one prompt instruction for arithmetic direction in change/difference questions.
3. Add record-local evidence selection using keyword and numeric matching.
4. Add a bounded one-shot retry for refusal or suspicious outputs.
5. Leave annotation-format mismatches as documented residual errors.

The main design story is:

```text
Failure analysis showed that remaining errors are mostly evidence selection and arithmetic-direction issues.
The next improvement should therefore be a lightweight evidence/calculation layer, not a full RAG system, full DSL program generator, or complex agent.
```
