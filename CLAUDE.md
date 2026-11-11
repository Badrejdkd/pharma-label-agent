# Instructions for AI coding assistants

This file is read by Claude (and similar assistants) before working on this repository.

## Project

Agentic research assistant over a fixed corpus of drug labels (`corpus/`): FastAPI API + LangGraph
agent + in-memory hybrid search. Answers must be grounded in the corpus, cited, reviewed and
followed by a "not medical advice" disclaimer. See `README.md` for the architecture.

## Commands

```bash
uv sync                                          # install (Python 3.11, versions pinned in uv.lock)
uv run pytest                                    # all tests, offline
uv run uvicorn app.main:app --reload --port 8000 # local server
uv run python -m eval.run_eval                   # evaluation with the real model (needs .env)
docker build -t agent . && docker run -p 8000:8000 --env-file .env agent
```

## Rules

- Never commit `.env` or `assignment-pharma.pdf` (confidential). Never print or log the API key.
- Never hardcode model names or URLs: everything comes from `app/config.py` (environment variables).
- All LLM calls go through `app/llm.py`. Do not call the OpenAI SDK anywhere else.
- Tools in `app/agent/tools.py` are deterministic, never raise to the agent (errors become messages
  returned to the model) and always return the chunks they read, so citations can be verified.
- Anything that can be checked by code (citations, numeric thresholds, disclaimer) is checked by code,
  not by the LLM.
- Prompts live in `app/agent/prompts.py`, in English. Answers follow the language of the question.
- Comments and docstrings are in French; identifiers are in English.
- Every change comes with a test. Tests must run offline: use `FakeLLM` and `FakeEmbedder` from
  `tests/fakes.py`.
- Keep functions short, fully typed, and explainable line by line: the author must be able to
  justify every line in an interview.

## Workflow used on this project

Work in small steps. For each step: write the code and its tests, run `uv run pytest`, run the real
model with a script from `scripts/` when behaviour depends on the LLM, review the output, then commit
with a descriptive message.
