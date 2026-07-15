# ConvFinQA Report
## 1. Introduction
This project aims to build a prototype for answers conversational dialogue questions over the ConvFinQA dataset. This reports shows a natural development envolvement on this task using modern LLM (Large Language Model) advancements since the paper release. All the code and documentation are submitted along with this report, to help on reproducing the results as well as any further discussion.

The final implementation is a Typer CLI with interactive chat tool, data exploration, methodology, result analysis and future work. Detailed explaination and the choice of methods are presented in the following sections, and a discussion of strengths, limitations and future work are also shown in this report.

## 2. Data exploration
The provided dataset file (data/convfinqa_dataset.json) contains 3037 "train" and 421 "dev" samples, which contains conversational numerical reasoning questions over unstructured financial documents, containing datatables. It is a cleaner version of the data described in the ConvFinQA paper [1]. The data has Type I (Simple) and Type II (Hybrid) conversations and a pre-extracted structured text table that contains table rows with cell values already linearised from the source filling, so the system can read table content directly from JSON. 

Table 1 shows that the data contains more multi-turn conversations rather than single questions, with most records have between two and four turns. This supports the fact that the system needs to reference previous questions and answers, and the evaluation should consider turn level performance rather than only the overall accuracy.

<div align="center">

<table>
  <tr>
    <th>Split</th><th>1 turn</th><th>2 turns</th><th>3 turns</th><th>4 turns</th><th>5 turns</th><th>6 turns</th><th>7 turns</th><th>8 turns</th><th>9 turns</th>
  </tr>
  <tr>
    <td>dev</td><td>0</td><td>116</td><td>94</td><td>103</td><td>88</td><td>17</td><td>2</td><td>1</td><td>0</td>
  </tr>
  <tr>
    <td>train</td><td>6</td><td>742</td><td>657</td><td>837</td><td>561</td><td>167</td><td>50</td><td>15</td><td>2</td>
  </tr>
</table>

<p><strong>Table 1. Dialogue length distribution aross train and dev.</strong></p>

</div>

The types of gold program in the data shows the numberical reasoning is built from a small set of arithmetic operations. Together, as what Table 2 shows, subtraction and division account for more than 70% of the total, which matches the nature of many financial questions: computing differences, ratios, margins, and percentage changes.

<div align="center">

<table>
  <tr>
    <th style="text-align:center;">Operation</th>
    <th style="text-align:center;">Count</th>
    <th style="text-align:center;">% of operation calls</th>
  </tr>
  <tr>
    <td>subtract</td>
    <td style="text-align:right;">5,131</td>
    <td style="text-align:right;">40.07%</td>
  </tr>
  <tr>
    <td>divide</td>
    <td style="text-align:right;">4,280</td>
    <td style="text-align:right;">33.42%</td>
  </tr>
  <tr>
    <td>add</td>
    <td style="text-align:right;">2,457</td>
    <td style="text-align:right;">19.19%</td>
  </tr>
  <tr>
    <td>multiply</td>
    <td style="text-align:right;">894</td>
    <td style="text-align:right;">6.98%</td>
  </tr>
  <tr>
    <td>greater</td>
    <td style="text-align:right;">40</td>
    <td style="text-align:right;">0.31%</td>
  </tr>
  <tr>
    <td>exp</td>
    <td style="text-align:right;">4</td>
    <td style="text-align:right;">0.03%</td>
  </tr>
</table>

<p><strong>Table 2. Gold program operation distribution aross train and dev.</strong></p>

</div>

As shown in the ConvFinQA paper, Table 3 summerises the main challenges that make ConvFinQA more difficult than simple questions answering. The main task requires the solutions to generate answers using the correct evidence, resolve conversational references across multiple turns, and perform accurate reasoning over many competing values [1]. These challenges motivates the version evolution: each version targets a different source of error, from evidence selection, pattern retrieval verification, and deterministic calculation execution.

<div align="center">

<table>
  <tr>
  </tr>
  <tr>
    <td>1. Many competing numeric candidates within each record</td>
  </tr>
  <tr>
    <td>2. Grounding the answer to the correct evidence or table row</td>
  </tr>
  <tr>
    <td>3. Selecting the correct value from nearby or similar candidates</td>
  </tr>
  <tr>
    <td>4. Resolving multi-turn references and follow-up questions</td>
  </tr>
  <tr>
    <td>5. Preventing later-turn error propagation</td>
  </tr>
  <tr>
    <td>6. Producing clean and parseable final answers</td>
  </tr>
  <tr>
    <td>7. Handling ratio and percentage scale consistently</td>
  </tr>
  <tr>
    <td>8. Choosing the correct operation, sign, and denominator</td>
  </tr>
  <tr>
    <td>9. Avoiding arithmetic mistakes in numerical reasoning</td>
  </tr>
  <tr>
    <td>10. Inferring the intended reasoning pattern for program-style questions</td>
  </tr>
</table>

<p><strong>Table 3. Key modeling challenges in ConvFinQA paper.</strong></p>

</div>

Our solution reads the provided ConvFinQA dataset from the given JSON file, where each record already contains pre-extracted document text, a structured table, dialogue questions, gold answers, and metadata. The structured table object is converted into a readable row-oriented table or evidence snippets with IDs (see more details in Section Methodology), which are used by different versions of our solution. Model outputs and evaluation results are save as JSONL file: batch runs write one prediction per turn, and the evaluation step later reads those saved predictions, compares them against `executed_answers`, and writes scored results back to JSONL. This separation makes the pipeline reproducible because the result generation and scoring can be rerun independently. 

In addition, a basic data leakage check was performed to understand how independent the `train` and `dev` splits are. We used LLM and tt founds that no exact same records are shared by both `train` and `dev`, but the data naturally contains repeated questions and similar reasoning patterns across different cases. Within each split, the `record_id` is unique, but there are some records were devrived from the same source PDF. As mentioned in the paper [1], this is expected in ConvFinQA because different conversations may come from the same financial page.

## 3. Methodology
### Scope and Evaluation Choices
Learned from the data and the scope of the task, full corpus-level RAG (Retrieval-Augmented Generation) is not used in our solution. The assignment data already provides the relevant `record_id` for each conversation and the central challenge is not retrieving the correct financial, but giving correct answer within the selected record. Using a full RAG pipeline with document indexing, chunk retrieval, vector search and re-ranking would add engineering complexity without providing direct benefit on solve the main challenges (shown as in Table 3) presented in this assignment.   

The evaluation focuses on `executed_answers` rather than `turn_program` because the goal of this prototype is to anwer the financial question correctly, rather than to reproduce the exact ConvFinQA program annotation. In addtion, `turn_program` only represents one possible reasoning program, different valid reasoning paths can also produce the same final answer, especially using a LLM-based system. So we use `executed_answers` for strict scoring, and the solution generates information similar to `turn_program` which is used to make the reasoning inspectable and to execute the final arithmetic deterministically. 

A fixed 500-record sample from the train data was used for development because full-train evaluation for testing every version would require significantly more LLM API calls, cost and runtime. The sample was selected using a fixed randome seed for reproducibility and was large enough to expose the main failure cases. The final evaluation is reported using the full dev data, which is kept separate for the main held-out comparision.

The solution is also integrated into the interactive chat tool. This means different versions can also be used and tested via the chat tool, rather than only through batch evaluation. Users can provide a `record_id` and the version number to see the actual answer to the question. It is useful to inspect the solution, especially the later version which fixes a verification or calulation issues that presented in the earlier versions.

### Iterative Solution Evolution
A single large design would make it unclear of which component or funtion was responsible for any regression of improvement. In this work, the solution was developed through controlled iterations, where each version adds one main capability which can be compared directly with the previous version. This makes the development process transparent, easier to show the thinking process behind the implementation, fits the scope of this task and better than a single all-in-one version. 

The first version `v1` uses the selected `record_id` and formats the entire record context, which includes the `pre_text`, `post_text`, the table content, conversation hitory and current question. The system then passes this context to the model with the conversation history. This `v1` provides a reference point that later version could add corresponding function that target the wrong evidence selection, incorrect value selection, arithmetic mistakes, messy final answers and multi-tun error propagation. 

```text
Record ID: Single_JKHY/2009/page_28.pdf-3

Pre-table text: ...

Table:
| metric | 2009 | 2008 | 2007 |
|---|---:|---:|---:|
| net cash from operating activities | 206588 | 181001 | 174247 |

Post-table text: ...

Conversation history:
(none)

Current question:
What was the net cash from operating activities in 2009?
```

The following tables show the results of running `v1` using the train sample. The challenges shown in the paper [1] can be grouped into a smaller set of observed error types (a LLM-assisted error analysis) which made the development priority clearer:

| Error type | Count | Related modeling challenges (shown in Table 3) | Development direction |
| --- | ---: | --- | --- |
| Calculation / program errors | 309 | Choosing the correct operation, sign, denominator; avoiding arithmetic mistakes; inferring reasoning pattern | Add verification, reasoning-pattern guidance, and later structured execution |
| Ratio or percentage errors | 231 | Handling ratio/percentage scale; choosing denominator; preventing arithmetic/format mismatch | Add percent/ratio normalization and denominator checks |
| Number-selection errors | 110 | Competing numeric candidates; grounding to the correct evidence/table row; selecting the correct nearby value | Add record-local evidence selection |
| Non-numeric / refusal errors | 65 | Producing clean parseable answers; avoiding unsupported refusals when numeric evidence exists | Add no-gold retry rules for refusal and final-answer format |
| Other / annotation ambiguity | 2 | Dataset/task-format mismatch | Document as limitation rather than overfit |

This analysis suggested two things. First, the largest source of error was Calculation / program errors , not only document retrieval. Second, grounding still had to come first because calculation cannot be correct if the wrong row or value is selected. Therefore the development order starts with a baseline, adds evidence grounding, verification/retry, then add reasoning-pattern retrieval, finally the deterministic execution of structured plans.

| Version | Main idea | Error target | Why it was introduced | LLM role |
| --- | --- | --- | --- | --- |
| `v1` | Full-record baseline | Baseline capability | Whether a modern LLM can answer from the full selected record plus conversation history. | Reads the full selected record, resolves the current turn, and produces the final answer. |
| `v2` | Record-local evidence selection | Number-selection errors | v2 highlights relevant text/table snippets before answering. | Reranks candidate snippets, then uses focused evidence plus the full record to answer questions. |
| `v3` | Evidence selection + verification retry | Non-numeric/refusal errors; ratio or percentage errors; simple calculation/program errors | v3 does a retry when local consistency checks detects likely failure. | Produces an initial answer, then revises once if the verifier flags a local inconsistency. |
| `v4` | v3 + train-example reasoning retrieval | Calculation/program errors, especially reasoning-pattern uncertainty | v4 retrieves similar solved train examples as operation-pattern hints, without using their numbers as evidence. | Uses similar solved examples as reasoning-pattern hints to anwswer the question given current value. |
| `v5` | v3 + structured calculation-plan execution | Calculation/program errors and ratio or percentage errors | v4 improved accuracy but added example noise and retrieval complexity. v5 uses an auditable plan and execute the arithmetic locally. | Extracts named values and proposes a calculation plan and uses local code executes the arithmetic. |

---

### Representative Version Improvements
We have tested `v1` using our train samples, and no suprise it did not produce perfect answers. 



rewrite this:
Few shot vs COT; mention no gold information is used. 


### Model Evaluation

rewrite this:
mention our solution vs the paper
show table of how much failure cases have been resolve by different cases


## 4. Engineering
gpt-4o-mini for cost and speed efficiency, workers for parallelisation. 
My system therefore focuses on selecting the relevant table rows and grounding extracted values to evidence IDs, rather than reconstructing table layout from the original document.
How our solution is connected to the chat tool.
workers.

## 5. Future Work 

Re-write this:
A natural future extension is to add structured conversation-state memory on top of v5. The current system passes previous turns back as compact text, such as Final answer: 4.7, which is useful but still leaves reference resolution to the LLM. Since v5 already produces named values, evidence IDs, and calculation steps, a future v6 could persist these outputs as structured state, for example drawn_amount = 4.7 from T-33 or facility_amount = 150 from T-31. Later follow-up questions such as “what percentage did that amount represent?” could then reuse these explicit variables instead of relying only on natural-language history. This would directly target one of the main remaining limitations: ambiguous multi-turn state tracking.

Re-write this:
dedicate table format

## Appendix
AI usage in this report: codex
Mentioned readme
Use reference in the paper to show the pain point

Rewrite this:

also avoided over-engineering because this is a 7-day prototype assignment, and the goal is to demonstrate disciplined modeling decisions rather than build the largest possible architecture. Since the dataset already provides the selected record, heavier designs such as corpus-level RAG, vector databases, multi-agent orchestration, or a full reimplementation of the paper’s DSL would add latency, complexity, and additional failure points without necessarily improving the core metric. Instead, I used a staged versioned design where each version targets an observed failure mode: v1 establishes the baseline, v2 improves grounding, v3 adds no-gold verification, v4 tests reasoning-pattern retrieval, and v5 adds deterministic execution for common arithmetic. This keeps the system easier to inspect, test, and compare while still making meaningful progress on the main ConvFinQA challenges.

A graph representation could be a useful future extension: records, table cells, dialogue turns, extracted values, and calculation steps could be represented as nodes and edges. This may improve table-cell grounding and multi-turn state tracking. However, it was not used in this prototype because the assignment already provides the selected record, and a graph database would add substantial engineering complexity beyond the core modeling challenges targeted in this submission.

## Reference
[1] Chen, Zhiyu, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. "Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering." In Proceedings of the 2022 conference on empirical methods in natural language processing, pp. 6279-6292. 2022.

[2] OpenAI. (2024, July 18). GPT-4o mini: Advancing cost-efficient intelligence. https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/