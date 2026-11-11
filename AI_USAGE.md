# How AI assistance was used

The assignment encourages AI assistance and asks for transparency. This file describes what the
assistant did, what I did, and how the result was checked.

## Assistant and set-up

- **Assistant:** Claude (Anthropic), used in a claude.ai Project dedicated to this assignment.
- **Access:** the assistant could read the project folder through the Claude desktop app, to inspect
  the corpus structure and check my files. It wrote the longest files (web interface, documentation,
  evaluation script) directly into the folder; I reviewed them before committing. It ran experiments
  in its own sandbox: unit tests with fake models, a browser check of the web interface, a dry run of
  the Docker steps.
- **Instructions:** the conventions it followed are in [CLAUDE.md](CLAUDE.md). No other skills or MCP
  servers were configured for this repository.

## How we worked

1. **Specification.** The assistant analysed the assignment PDF and the corpus, then drafted a
   specification (requirements, tools, architecture, evaluation plan, schedule, risks). I chose the
   stack from its proposals.
2. **Step-by-step build.** The project was built in nine steps, each one a commit: project set-up,
   configuration and LLM client, corpus ingestion, hybrid index, tools, LangGraph agent, API and web
   interface, Docker, evaluation and documentation. For each step the assistant proposed the code and
   its tests; I created the files, ran the tests and scripts on my machine, and sent back the output.
3. **Checks against reality.** Several decisions came from real runs rather than assumptions:
   - a capability test of the provided model (`scripts/check_llm.py`) before writing the agent;
   - the typo `OPENAI_BASE_UR` in the provided `.env`, and quotes kept by `docker run --env-file`;
   - retrieval alone missing the warfarin table that lists sertraline, which led to the
     `find_interactions` tool and its drug-class matching;
   - the agent over-fetching evidence (12 tool calls for one question), fixed in the research prompt.
4. **Review.** I read every file before committing it and asked for explanations whenever a part was
   unclear. I am accountable for all the code in this repository.

## What I would do differently with more time

Use a different model for the critic than for the writer, add an LLM judge to the evaluation, and
compare prompt variants on the evaluation set before choosing one.
