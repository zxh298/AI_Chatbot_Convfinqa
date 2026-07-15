# ConvFinQA Assignment
## See REPORT.md for the complete report of this assignment
## Get started
### Prerequisites
- Python 3.13.13
- [UV environment manager](https://docs.astral.sh/uv/getting-started/installation/)

### Setup
1. Clone this repository
2. Use the UV environment manager to install dependencies:

```bash
# install uv
brew install uv

# set up env
uv sync
```

3. Create a local `.env` file in the root directory with:

``` 
OPENAI_API_KEY=your_api_key_here
```


Do not commit `.env`, because it contains local secrets.

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

Run the baseline full-record version:

```bash
uv run main chat Single_PNC/2015/page_48.pdf-1 --version v1
```

Available answer versions:

- `v1`: full-record baseline
- `v2`: record-local evidence selection
- `v3`: evidence selection + no-gold verification retry
- `v4`: v3 + retrieved train-example reasoning guidance
- `v5`: v3 + structured calculation-plan execution
- `v5a`: v5 + limited offline numeric fallback

For versions that use evidence selection (except for `v1`), add `--show-evidence` to inspect selected snippets:

```bash
uv run main chat Single_PNC/2015/page_48.pdf-1 --version v2 --show-evidence
```

#### Run the model in batch

```bash
uv run main run \
  --split train \
  --model gpt-4o-mini \
  --max-records 500 \
  --random-seed 42 \
  --version v1 \
  --workers 2 \
  --output-path outputs/run_train_500_random42_gpt4o_mini_v1.jsonl
```
and on dev data:
```bash
uv run main run \
  --split dev \
  --model gpt-4o-mini \
  --version v1 \
  --workers 2 \
  --output-path outputs/run_dev_gpt4o_mini_v1.jsonl
```

Parameters:

- `--split train`: choose which dataset split to run (`train` or `dev`).
- `--model gpt-4o-mini`: choose the OpenAI model used for answer generation.
- `--max-records 500`: limit the run to 500 records from the input data; omit this to run the full split.
- `--random-seed 42`: sample records reproducibly when `--max-records` is used.
- `--version v1`: select the answer pipeline version (`v1`, `v2`, `v3`, `v4`, `v5`, or `v5a`).
- `--workers 2`: run multiple records concurrently using 2 workers; turns inside each record still run sequentially.
- `--output-path ...jsonl`: save raw model predictions to a JSONL file.

This replays dataset `conv_questions` and writes one raw JSONL row per model answer. The run file stores predictions only; it does not store gold answers or correctness.

#### Evaluate saved predictions

This compares saved predictions against strict `executed_answers`, prints Table 4 / Figure 5-style breakdowns, and optionally writes a scored JSONL file.

```bash
uv run main evaluate \
  outputs/run_train_500_random42_gpt4o_mini_v1.jsonl \
  --output-path outputs/eval_train_500_random42_full_gpt4o_mini_strict_executed.jsonl
```

on dev data:
```bash
uv run main evaluate \
  outputs/run_dev_gpt4o_mini_v1.jsonl \
  --output-path outputs/eval_dev_gpt4o_mini_v1_strict_executed.jsonl
```

#### AI usage in this work
AI tools were used during this work as an implementation assistant. Codex was used to help to draft and edit Python modules, generate unit tests, run evaluation commands, format Markdown tables, create README file, and summarize experimental results. The methodology, version design, interpretation of results, project scope, and final submitted report were directed and created by the author.
