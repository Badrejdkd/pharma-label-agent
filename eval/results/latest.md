# Evaluation results

Target: in-process agent. Date: 2026-11-11 00:59.

| Metric | Value |
| --- | --- |
| Questions | 12 (1 errors) |
| Expected sources cited | 100% |
| Key facts present | 91% |
| Disclaimer present | 100% |
| Critic verdicts | pass: 9, revised: 1, flagged: 1 |
| Latency (median / max) | 17.2 s / 31.3 s |

| Id | Sources | Facts | Verdict | Tools chosen by the agent | Latency |
| --- | --- | --- | --- | --- | --- |
| public-1 | 100% | 100% | pass | get_section, get_section, search_corpus | 19.6 s |
| public-2 | 100% | 100% | revised | compare_sections | 31.3 s |
| public-3 | 100% | 100% | pass | find_interactions, search_corpus, search_corpus | 17.2 s |
| public-4 | 100% | 100% | pass | check_lab_threshold | 9.7 s |
| extra-interaction | 100% | 100% | pass | find_interactions, search_corpus, search_corpus | 18.1 s |
| extra-mechanism | 100% | 100% | pass | get_section | 11.2 s |
| extra-max-dose | error: Appel au LLM impossible : Error code: 400 - litellm.BadRequestError: AzureExcept | | | | 6.1 s |
| extra-contrast | 100% | 100% | pass | search_corpus, search_corpus | 16.2 s |
| extra-allergy | 100% | 100% | pass | get_section | 20.7 s |
| extra-aggregation | 100% | 100% | pass | search_corpus | 11.1 s |
| extra-out-of-corpus | 100% | 0% (missing: `not (covered\|included\|in the corpus\|available)\|does not (cover\|contain\|include)\|no (FDA )?label\|only .*glossary`) | flagged | get_section, search_corpus | 16.6 s |
| extra-french | 100% | 100% | pass | search_corpus, get_section | 19.5 s |
