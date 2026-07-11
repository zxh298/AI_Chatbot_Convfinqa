# ConvFinQA Assignment


Thank you for taking the time to do this assignment! Please see the [main Notion page](https://tomoroai.notion.site/Technical-Assignment-1fa0de3387ea80debb36cda4ae41e93d) for the full instructions. 


We have cleaned up the dataset; please see `dataset.md` for more information. We recommend you use this version of the data for the assignment, as it will save you a lot of time. If you have any questions, please don't hesitate to ask your point of contact. 


Good luck! 

## Get started
### Prerequisites
- Python 3.12+
- [UV environment manager](https://docs.astral.sh/uv/getting-started/installation/)

### Setup
1. Clone this repository
2. Use the UV environment manager to install dependencies:

```bash
# install uv
brew install uv

# set up env
uv sync

# add python package to env
uv add <package_name>
```

### Run the ConvFinQA CLI

The project exposes a [Typer](https://typer.tiangolo.com/) CLI with three workflows: interactive chat, batch evaluation, and offline result analysis.

Show the available commands:

```bash 
uv run main
```

You can also use the longer form:

```bash
uv run python src/main.py
```

Before calling the OpenAI API, set `OPENAI_API_KEY` in your environment or `.env` file.

#### Chat with one record

```bash
uv run main chat Single_PNC/2015/page_48.pdf-1
```

This loads the selected ConvFinQA record, sends its `pre_text`, table, `post_text`, and conversation history to the model, and lets you ask follow-up questions interactively.

To enable the record-local evidence-selection version, add `--use-evidence`:

```bash
uv run main chat Single_PNC/2015/page_48.pdf-1 --use-evidence
```

To inspect which snippets were selected on each turn, add `--show-evidence`:

```bash
uv run main chat Single_PNC/2015/page_48.pdf-1 --use-evidence --show-evidence
```

[![Chat](figures/chat_example.png)](figures/chat.png)  

#### Run batch evaluation

```bash
uv run main eval-baseline \
  --split train \
  --model gpt-4o-mini \
  --max-records 500 \
  --random-seed 42 \
  --output-path outputs/eval_train_500_random42_full_gpt4o_mini_strict_executed.jsonl
```

This replays dataset `conv_questions`, compares parsed model answers against strict `executed_answers`, and writes one JSONL row per evaluated turn.

To evaluate the evidence-selection version, add `--use-evidence` and write to a separate output file:

```bash
uv run main eval-baseline \
  --split train \
  --model gpt-4o-mini \
  --max-records 500 \
  --random-seed 42 \
  --use-evidence \
  --output-path outputs/eval_train_500_random42_full_gpt4o_mini_evidence.jsonl
```

#### Analyze saved results

```bash
uv run main analyze-results outputs/eval_train_500_random42_full_gpt4o_mini_strict_executed.jsonl
```

This reads a saved JSONL file and prints Table 4 / Figure 5-style breakdowns without making additional API calls.

## Submission 
Please make a submission branch & make a PR to main. The PR should contain: 


- A solution to the main task
- A report summarising your findings. We have sketched out a template for you in `REPORT.md`, but you can use any other setup (like LaTeX) if you prefer.
- Please send a link to the PR to [recruitment@tomoro.ai](mailto:recruitment@tomoro.ai) with the subject `submission: <your name>`.
  
NOTE: Please DO NOT merge any of your submission to main, all of your work should be on your branch `submission`. 


**Please let us know if you used any AI tools to help generate code for your assignment.**
Using AI-powered IDEs or coding assistants is acceptable, as these are commonly used in real-world environments, and this assignment is intended to reflect that. If you’ve used AI tools to help you write code or your report, or any other part of your process, we ask that you disclose how and where you used them. This isn’t to catch you out. It’s an opportunity to show that you understand how to use these tools effectively and responsibly as part of your workflow.
