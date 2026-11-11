# Evaluation results

Target: in-process agent. Date: 2026-11-11 13:26.

| Metric | Value |
| --- | --- |
| Questions | 12 (0 errors) |
| Expected sources cited | 100% |
| Key facts present | 92% |
| Disclaimer present | 100% |
| Critic verdicts | pass: 11, flagged: 1 |
| Latency (median / max) | 14.25 s / 20.7 s |

| Id | Sources | Facts | Verdict | Tools chosen by the agent | Latency |
| --- | --- | --- | --- | --- | --- |
| public-1 | 100% | 100% | pass | get_section | 14.8 s |
| public-2 | 100% | 100% | pass | compare_sections | 16.5 s |
| public-3 | 100% | 100% | pass | find_interactions | 11.9 s |
| public-4 | 100% | 100% | pass | check_lab_threshold | 9.4 s |
| extra-interaction | 100% | 100% | pass | find_interactions, search_corpus, search_corpus | 19.1 s |
| extra-mechanism | 100% | 100% | pass | get_section | 13.7 s |
| extra-max-dose | 100% | 100% | pass | get_section, get_section, search_corpus | 18.4 s |
| extra-contrast | 100% | 100% | pass | get_section | 12.6 s |
| extra-allergy | 100% | 100% | pass | get_section, search_corpus | 13.1 s |
| extra-aggregation | 100% | 100% | pass | get_section | 13.0 s |
| extra-out-of-corpus | 100% | 0% (missing: `no (passage\|label\|information\|section)\|not (covered\|included\|in the corpus\|available\|found)\|does not (cover\|contain\|include)\|cannot be (established\|determined\|answered)\|only .*glossary`) | flagged | get_section | 17.2 s |
| extra-french | 100% | 100% | pass | search_corpus, get_section | 20.7 s |
