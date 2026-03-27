# Prompt From Examples

This project turns a dataset of input/output examples into a reusable few-shot prompt.

The first iteration intentionally stays simple:

- Step 1: a small, hand-written prompt plus a compact few-shot file and tests
- Step 2: a larger dataset file that acts like a prompt corpus
- Step 3: a real Python package with linting, formatting, env support, CLI entrypoints, and tests
- Step 4: a prompt optimizer that can work from a raw user request, or infer the task from the dataset when no prompt is given
- Step 5: an eval harness that compares the developer baseline prompt A against the production-generated prompt B

## Chosen Demo Task

The demo task is support ticket triage. Each free-form user message is normalized into JSON with:

- `category`: `billing`, `bug`, `feature_request`, `account_access`
- `priority`: `low`, `medium`, `high`
- `language`: `tr`, `en`

This keeps the prompt problem small enough to move fast, while still being realistic enough to exercise few-shot selection.

## Prompt Roles

- Prompt A: a developer-authored baseline prompt used only for internal comparison and regression testing
- Prompt B: the production prompt created by the tool from a rough user intent and/or the dataset

The current goal is to make prompt B at least as strong as prompt A on a held-out evaluation set.

## Project Layout

- `prompt_datasets/support_triage_fewshot_small.json`: the small few-shot examples for the first prompt
- `prompt_datasets/support_triage_dataset.json`: the larger dataset used by the optimizer
- `prompt_datasets/support_triage_eval.json`: the held-out eval set used for A vs B comparisons
- `src/prompt_from_examples/baseline.py`: the initial hand-authored prompt
- `src/prompt_from_examples/evals.py`: prompt execution and comparison logic
- `src/prompt_from_examples/selection.py`: relevance-based few-shot example selection
- `src/prompt_from_examples/task_inference.py`: task inference from raw user intent and/or the dataset
- `src/prompt_from_examples/optimizer.py`: the prompt builder that combines inference and example selection
- `tests/`: baseline, dataset, selection, and optimizer tests

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
Copy-Item .env.example .env
```

## Commands

Render the small baseline prompt:

```powershell
prompt-from-examples baseline
```

Build an optimized prompt from the large dataset and a rough human request:

```powershell
prompt-from-examples optimize --user-intent "Turn messy support requests into stable JSON labels for the ops team"
```

Build an optimized prompt without providing any task description:

```powershell
prompt-from-examples optimize
```

Compare prompt A and prompt B with the offline simulator:

```powershell
prompt-from-examples compare --show-failures
```

If `OPENAI_API_KEY` is present, you can let the app refine the inferred instruction with an LLM:

```powershell
prompt-from-examples optimize --use-openai
```

Run the same A vs B comparison with a live OpenAI backend once a key is available:

```powershell
prompt-from-examples compare --eval-backend openai --use-openai-inference
```

## Quality Checks

```powershell
python -m pytest
python -m ruff check .
python -m black --check .
```

## How The Optimizer Works

1. Load the dataset examples.
2. Infer the task instruction from the dataset schema and the optional raw user request.
3. Select the most relevant examples from the larger dataset.
4. Render a final prompt with a stable output contract and the chosen few-shot block.

## How The Eval Harness Works

1. Build prompt A from the small developer-owned baseline examples.
2. Build prompt B from the production dataset and the optional user intent.
3. Run both prompts on the same held-out eval dataset.
4. Compare exact-match, category, priority, and language accuracy.

## Notes

- The default inference engine is heuristic so the repository works offline and stays fully testable.
- The offline `compare` command uses a deterministic simulator so the repo can test prompt regressions without an API key.
- The OpenAI-backed inference and eval paths are optional and only used when you explicitly enable them.

## Next Steps

- Let the tool propose new few-shot examples automatically when the dataset is too thin.
- Add live prompt evaluations against a real LLM and score prompt variants on a validation split.
- Replace the simple token-overlap selector with embeddings or a reranker.
