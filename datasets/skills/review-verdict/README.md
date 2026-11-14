# Static review verdict pilot

This fixture adapts the independent reviewer prompt from Hermes Agent's `requesting-code-review` skill. It covers the code-review target described in this repository's Phase 1 plan. The checkout contains no populated datasets or bundled target skill.

Source: [requesting-code-review at e05c8d578c6f4f919b4a2a4c308733144ac4f8b0](https://github.com/NousResearch/hermes-agent/blob/e05c8d578c6f4f919b4a2a4c308733144ac4f8b0/skills/software-development/requesting-code-review/SKILL.md), lines 137-175. The source declares MIT licensing and attribution to Hermes Agent, adapted from obra/superpowers and MorAlekss. The fixture preserves the verdict policy and security categories. It replaces tool-based verification with an explicit single-turn static-review boundary. It does not reproduce the complete skill workflow.

The handwritten cases contain synthetic diffs. Each lineage occurs in exactly one split. The optimizer receives training rubrics and observed development feedback. The task model receives the skill and task input, never an expected verdict or rubric.

`expected_behavior` supplies the repository's keyword-overlap fitness. `expected_passed` supplies a known-answer verdict label for a separate audit. Verdict accuracy does not prove that every finding is correct. Neither metric proves that an evolved skill improves a live Hermes agent.

Holdout labels are public for reproducibility. The experiment runner excludes them from every search interface and audits them only after all search decisions are sealed. This is an internal protocol boundary, not a cryptographic secret.
