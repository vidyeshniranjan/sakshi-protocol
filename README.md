# Sakshi-Protocol

**An epistemic control layer that reduces hallucination in large language models by verifying claims against external sources and detecting when a model is reasoning beyond what it actually knows.**

Model-agnostic. No retraining. Wraps around an existing LLM as a control layer.

📄 Preprint: [Zenodo DOI 10.5281/zenodo.20621587](https://doi.org/10.5281/zenodo.20621587) · ORCID [0009-0007-4186-2925](https://orcid.org/0009-0007-4186-2925)

\---

## What it does

Large language models produce fluent text whether or not the underlying claim is true, because generation and verification happen inside the same probabilistic step. Sakshi-Protocol separates the two. It sits around the model and, for every response, runs a control loop that decides whether to **accept**, **retrieve and ground**, or **abstain**.

It does this through:

* A **claim extraction and verification** path that checks generated claims against retrieved evidence using natural-language inference.
* An **absence-detection** path that recognises unanswerable or out-of-scope questions instead of fabricating an answer.
* An **epistemic scoring** module that produces a distortion signal when the model's confidence is not supported by evidence, with **per-model calibrated thresholds** rather than a single privileged reference model.
* An optional **grounding mode (Sakshi+Ω)** that retrieves external evidence on the subset of prompts that need it.

\---

## Results

Evaluated on a 500-prompt benchmark across three models in three configurations (baseline, Sakshi, Sakshi+Ω). The headline metric is the **hallucination-intervention rate**: on the hallucination-prone prompt class (Class A), how often the system correctly intervenes (abstains or grounds) instead of emitting a confident false answer, while preserving accuracy on ordinary prompts.

|Model|Mode|Hallucination intervention (Class A)|Overall accuracy|
|-|-|-|-|
|Claude Sonnet 4.6|baseline|0%|98.7%|
|Claude Sonnet 4.6|**Sakshi+Ω**|**100%**|98.0%|
|Llama 3.3 70B|baseline|0%|96.7%|
|Llama 3.3 70B|**Sakshi+Ω**|**98%**|92.0%|
|Qwen 3.5 9B|baseline|0%|98.0%|
|Qwen 3.5 9B|**Sakshi+Ω**|**100%**|95.3%|

Baseline models almost never flag their own hallucinations (0% intervention). With Sakshi+Ω, 98–100% of hallucination-prone prompts are intervened on, while overall accuracy on normal prompts stays high. Full per-mode numbers are in [`results/metrics.json`](results/metrics.json).

!\[Hallucination intervention](results/plots/fig2\_hallucination\_intervention.png)

\---

## How it works

```
Prompt
  │
  ▼
Generator (any LLM)  ──►  candidate response
  │
  ▼
Observer  ──►  signals: uncertainty, grounding, consistency
  │
  ▼
Epistemic scorer  ──►  distortion D (per-model calibrated thresholds)
  │
  ▼
Controller  ──►  accept  │  retrieve + ground (Ω)  │  abstain
```

* `src/sakshi/controller.py` — the accept/retrieve/abstain decision and per-model thresholds
* `src/sakshi/pipeline.py` — end-to-end orchestration
* `src/sakshi/epistemic\_scorer.py` — distortion signal
* `src/sakshi/claim\_extractor.py`, `claim\_verifier.py` — claim-level verification
* `src/sakshi/abstention.py` — absence-detection paths
* `src/sakshi/retriever.py` — grounding retrieval for Ω mode

\---

## Quickstart

```bash
git clone https://github.com/vidyeshniranjan/sakshi-protocol.git
cd sakshi-protocol
python -m venv .venv \&\& source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env        # add your API keys
```

Run the benchmark on a single model with the bundled sample prompts:

```bash
python benchmarks/run\_benchmark.py \\
  --model claude-sonnet-4-6 \\
  --mode sakshi \\
  --prompts prompts/sample\_prompts.json

```

Keys are loaded automatically from your `.env` file (or from environment variables if set). Nothing is hardcoded.

**API keys required:**

* `OPENAI\_API\_KEY` — **required.** Used for embeddings (the semantic-similarity signal) on every prompt, and for reasoning-consistency checks on reasoning-type prompts.
* `ANTHROPIC\_API\_KEY` — required when running `--model claude-sonnet-4-6`.
* `TOGETHER\_API\_KEY` — required when running `--model llama-3.3-70b` or `--model qwen-3.5-9b`.

So a typical run needs **two** keys: `OPENAI\_API\_KEY` plus the key for whichever generation model you select.

\---

## Scope and honesty notes

* The intervention figures are measured **on this benchmark set**, not a universal guarantee. The benchmark deliberately oversamples hallucination-prone prompts.
* Thresholds are calibrated per model family because distortion-signal distributions differ across models. This is a property of the method, not a workaround.
* Standard external benchmarks (TruthfulQA, HaluEval) and learned calibration are V4.0 scope.

\---

## Citation

```bibtex
@misc{vidyesh2026sakshi,
  author       = {Vidyesh, N. K.},
  title        = {Sakshi-Protocol: Epistemic Control for Hallucination
                  Mitigation in Large Language Models},
  year         = {2026},
  doi          = {10.5281/zenodo.20621587},
  url          = {https://doi.org/10.5281/zenodo.20621587}
}
```

## License

MIT. See [LICENSE](LICENSE).

