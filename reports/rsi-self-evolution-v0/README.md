# Three-seed static-review pilot

All nine runs completed with equal task-call budgets. Guided selection did not establish a skill-quality gain. No candidate is promoted or deployed.

| Arm | Mean final keyword fitness | Mean final verdict accuracy |
|---|---:|---:|
| Native GEPA text adapter | 0.579109 | 62.50% |
| Prediction-guided | 0.581944 | 58.33% |
| Random selection | 0.572627 | 62.50% |

Guided-minus-GEPA keyword fitness is +0.002836, while verdict accuracy is lower by 4.17 percentage points. Guided-minus-random keyword fitness is +0.009317, with the same accuracy deficit. These tiny samples do not support a significance or generalization claim. Selected-only predictor correlations are weak, noisy, and biased.

Each arm has 28 development and 32 holdout task calls per seed. The complete pilot has 252 development, 288 holdout, and 31 generation calls: 571 successful requests, 270,465 prompt tokens, 90,357 completion tokens, and zero reasoning tokens. Task execution uses Nemotron Lightning; mutation uses Nemotron Ultra, both non-thinking.

See [aggregate rows, model roles, and paired deltas](pilot-v3/summary.json), [fixed-checkpoint curves](pilot-v3/rounds.csv), and [actual-receipt verification](pilot-v3/verification.json). The earlier [invalid Groq run](groq-invalid/summary.json) and [interrupted Lightning run](nim-interrupted/summary.json) are not quality comparisons.

The executed source is `0c46a34b7b821f20c85f7f72aa38dbad5550c128`. Every search is sealed before any holdout call. Repeated audits of unchanged prompts still vary, and holdout results never select a winner. The primary metric is keyword overlap; verdict accuracy is a handwritten static fixture audit, not full-agent verification. The predictor is a sparse embedding regressor, not JEPA. The original evolution CLI and live skills remain unchanged.

An independent Nemotron Ultra evidence audit found no evidenced inconsistencies in the supplied code, aggregates, and conclusions. Reviewer agreement does not replace receipt verification or establish task quality.

[Reproduction command and implementation details](../../docs/rsi-anything.md). Publication study: [draft PR](https://github.com/kvnloo/z0evals/pull/190).
