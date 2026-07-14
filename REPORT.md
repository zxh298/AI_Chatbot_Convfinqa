# ConvFinQA Report
## Introduction
This project aims to build a prototype for answers conversational dialogue questions over the ConvFinQA dataset. This reports shows a natural development envolvement on this task using modern LLM advancements since the paper release. All the code and documentation are submitted along with this report, to help on reproducing the results as well as any further discussion.

The final implementation is a Typer CLI with interactive chat tool, data exploration, methodology, result analysis and future work. Detailed explaination and the choice of methods are presented in the following sections, and a discussion of strengths, limitations and future work are also shown in this report.

## Data exploration
The provided dataset file (data/convfinqa_dataset.json) contains 3037 "train" and 421 "dev" samples, which contains conversational numerical reasoning questions over unstructured financial documents, containing datatables. It is a cleaner version of the data described in the ConvFinQA paper [1]. The data has Type I (Simple) and Type II (Hybrid) conversations and a pre-extracted structured text table that contains table rows with cell values already linearised from the source filling, so the system can read table content directly from JSON. 

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

<p><strong>Table 1. Dialogue length distribution by split.</strong></p>

</div>


reference the pain point in the paper

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
