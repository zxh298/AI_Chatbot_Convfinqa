# ConvFinQA Report
## Introduction
This project aims to build a prototype for answers conversational dialogue questions over the ConvFinQA dataset. This reports shows a natural development envolvement on this task using modern LLM advancements since the paper release. All the code and documentation are submitted along with this report, to help on reproducing the results as well as any further discussion.

The final implementation is a Typer CLI with interactive chat tool, data exploration, methodology, result analysis and future work. Detailed explaination and the choice of methods are presented in the following sections, and a discussion of strengths, limitations and future work are also shown in this report.

## Data exploration
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

The tyoes of gold program in the data shows the numberical reasoning is built from a small set of arithmetic operations. Together, as what Table 2 shows, subtraction and division account for more than 70% of the total, which matches the nature of many financial questions: computing differences, ratios, margins, and percentage changes.

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


## Method
The ConvFinQA paper follows the same metric as in FinQA, the execution accuracy to evaluate the final execution result and program accuracy to evaluate program equivalence [1]. 

Data engineering: table parser
why not RAG


### Engineering
gpt-4o-mini for cost and speed efficiency, workers for parallelisation. 
My system therefore focuses on selecting the relevant table rows and grounding extracted values to evidence IDs, rather than reconstructing table layout from the original document.
## Error Analysis
## Future Work 
## Abstract
AI usage in this report: codex
## Mentioned readme
Use reference in the paper to show the pain point

## Reference
[1] Chen, Zhiyu, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. "Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering." In Proceedings of the 2022 conference on empirical methods in natural language processing, pp. 6279-6292. 2022.
