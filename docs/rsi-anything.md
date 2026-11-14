# Prediction-guided skill search

The experimental optimizer evaluates a baseline, proposes several candidate bodies, predicts their development scores, evaluates only the selected candidates, and updates a surrogate from real outcomes. The random control uses the same candidate pool size and evaluation allowance. The native GEPA control uses the repository's GEPA search engine through an explicit text adapter.

This is an RSI-inspired adaptation, not a reproduction of JEPA-Anything or evidence of sustained recursive self-improvement.

## Run the comparison

1. Install the experiment dependencies with `python -m pip install -e '.[dev,rsi]'`.
2. Supply `NVIDIA_API_KEY` through your secret manager for the measured configuration below. Groq requires `GROQ_API_KEY` instead. Do not put a key in a command or file.
3. Set `RUN_OUTPUT` to a new directory outside the checkout for private raw receipts.
4. Run the comparison.

```bash
python -m evolution.skills.compare_optimizers \
  --skill-path datasets/skills/review-verdict/SKILL.md \
  --dataset datasets/skills/review-verdict \
  --output "$RUN_OUTPUT" \
  --seeds 0,1,2 --rounds 3 --population-size 4 --top-k 2 \
  --task-lane nim-lightning --proposer-lane nim-ultra
```

The CLI defaults to `qwen/qwen3.8-27b` on Groq. The measured run overrides those defaults: Nemotron Lightning executes tasks and Nemotron Ultra proposes mutations. Lightning passed task probes but repeatedly failed exact-count mutation probes; Ultra passed both one- and four-candidate preflights. The OpenAI-compatible client is a transport library, not a model under test.

Both Nemotron lanes use the vendor's documented `enable_thinking=False` mode. Task responses and proposal responses use JSON-object mode. Generation reserves 3,072 output tokens and uses at most two truncated development-feedback records. These bounds keep the small pilot within provider payload limits.

The runner refuses an existing output directory. Successful calls share a provider-level pacing clock. When headers expose a token-per-minute limit, the client targets at most 75 percent of it. Receipts retain safe quota headers and pacing durations. A failed request never receives a fabricated fitness score. It records invalid arms and returns a nonzero exit status when any arm is incomplete. HTTP 401, 429, and transient server errors receive at most three attempts with exponential delays. A persistent authentication or rate-limit failure closes that client lane for the rest of the run. A hard cap limits each provider to 2,900 attempts per process.

## Evaluation controls

The fixture has four training cases, four validation cases, and eight holdout cases. Each problem lineage occurs in one split. All arms start from the same skill and data. Arm order changes deterministically with the seed.

With the default settings, every arm has 28 task-model selection calls, including the baseline and GEPA's training evaluations. A hard boundary stops GEPA even if its internal batch would exceed that allowance. Incomplete batches do not become successful candidates. The guided and random arms evaluate seven complete four-case validation batches.

Every arm receives four post-search audits of eight holdout cases, for 32 holdout calls. The checkpoints correspond to selection-call counts 4, 12, 20, and 28. For GEPA, each checkpoint uses the best fully validated candidate available at that count. An unchanged incumbent may appear more than once.

All arms and seeds finish search before the runner writes `search-seal.json` and starts any holdout call. Search-file hashes protect the audit boundary. Holdout scores never affect generation, predictor updates, or retention. Request attempts, retries, generation calls, task calls, tokens, and latency remain separate. Recovered retries cannot silently increase a valid arm's task allowance.

The primary metric reuses `skill_fitness_metric`, which measures keyword overlap. The separate known-answer verdict audit checks the response's `passed` field and basic JSON consistency against handwritten labels. Neither check validates every finding or proves that a live Hermes workflow improved.

Spearman correlation pairs pre-update predictions with selected candidates' real development scores. It uses average ranks for ties and returns null for undefined constant vectors. This selected-only statistic is biased and small. It does not estimate full-pool ranking quality.

## Baseline compatibility

The original CLI calls `dspy.GEPA(max_steps=...)`, which current DSPy rejects. Its `skill_text` is an input rather than an optimizable parameter. Its structure gate also receives a bare body instead of a full skill file. These issues appear in upstream issues [10](https://github.com/NousResearch/hermes-agent-self-evolution/issues/10), [141](https://github.com/NousResearch/hermes-agent-self-evolution/issues/141), and [199](https://github.com/NousResearch/hermes-agent-self-evolution/issues/199).

The experiment does not silently repair or replace that CLI. Its baseline uses native `gepa.optimize` with explicit candidate bodies and `max_metric_calls`. Proposal generation, keyword fitness, and artifact constraints use the same boundaries as the guided and random arms. GEPA retains its own Pareto selection and minibatch acceptance. This is a working compatibility adaptation, not a literal run of the original wrapper.

Candidate generation cost is not equal across arms. The guided and random arms generate four variants per round. GEPA generates one mutation per reflective step. Equal external task calls do not imply equal tokens, latency, or search distributions.

## Predictor and paper deviations

The surrogate uses normalized sparse word embeddings and cosine-kernel regression over the five nearest observed artifacts. It has no pretrained semantic encoder, learned latent dynamics, EMA target, or OPF factors. Each arm and seed receives fresh predictor state. Updates contain real development outcomes only.

The native paper code uses a Qwen2.5-Coder-3B bfloat16 encoder with LoRA, a target encoder, OPF factors, and execution readouts. Online adaptation needs substantially more memory than the base weights alone. The pilot avoids allocating that model on a shared GPU and does not claim JEPA equivalence.

The paper uses reliability-triggered predictor updates and component/pair retention checks. This pilot updates its nonparametric regressor after every measured round and retains skills by validation keyword score. It omits the paper's predictor-quality and policy-model compatibility gates. No candidate is deployed.

Sources:
- [RSI-Anything paper, arXiv:2610.13912](https://arxiv.org/abs/2610.13912).
- [RSI-Anything at 8ba44cd73ec6f57b151ec06e2f71a0726adfb687](https://github.com/Gen-Verse/RSI-Anything/tree/8ba44cd73ec6f57b151ec06e2f71a0726adfb687).
- [JEPA-Anything at c6e6c88f3ef75a4ce7acd660d6fa5779d995512c](https://github.com/Gen-Verse/JEPA-Anything/tree/c6e6c88f3ef75a4ce7acd660d6fa5779d995512c).
- [Released RSI resources](https://huggingface.co/datasets/Gen-Verse/RSI-Anything).
