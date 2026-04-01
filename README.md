# Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents

> **Paper**: [Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents](https://arxiv.org/abs/2603.26233)
>
> This repository contains the code and evaluation setup for the paper above. We develop and evaluate uncertainty-aware clarification-seeking agents on an underspecified variant of SWE-bench Verified ([Vijayvargiya et al., 2026](https://arxiv.org/abs/2502.13069)), where agents must independently decide when to ask the user clarifying questions to resolve missing information. We use the [OpenHands](https://github.com/All-Hands-AI/OpenHands) agent framework for agent development and testing.

## 📋 Overview

We evaluate five experimental settings on the SWE-bench Verified dataset, using **Claude Sonnet 4.5** as the coding agent backbone and **GPT-5.1** as the simulated user:

| Setting | Paper name | Description |
|---|---|---|
| `full` | **Full** | Agent receives the fully specified original GitHub issue. No user interaction. Upper-bound baseline. |
| `hidden` | **Hidden** | Agent receives the underspecified issue. No user interaction. Lower-bound baseline. |
| `interact` | **Interactive Baseline** | Agent receives the underspecified issue and is explicitly instructed to query the user before proceeding. Hardcoded single interaction turn. |
| `clarify_interact_v2` | **UA-Single** | Agent receives the underspecified issue. A single coding agent is reminded each turn to assess for underspecification and query the user if needed. |
| `clarify_interact` | **UA-Multi** | Agent receives the underspecified issue. A dedicated Intent Agent monitors each turn for underspecification and constrains the Main Agent to query the user when needed. |

### 📊 Key results

| Setting | Resolve rate |
|---|---|
| Full | 70.80% |
| Hidden | 54.80% |
| Interactive Baseline | 70.40% |
| UA-Single | 61.20% |
| **UA-Multi** | **69.40%** |

Our UA-Multi agent closes the performance gap with agents operating on fully specified instructions, achieving a resolve rate closely matching Full (p = 0.458) and Interactive Baseline (p = 0.621), while significantly outperforming our UA-Single agent (p < 0.001). All p-values are computed via non-parametric permutation tests.

## ⚙️ Setup

Install OpenHands by following the [OpenHands installation guide](https://docs.all-hands.dev/usage/installation), then additionally configure your LLM settings in `config.toml`.

### LLM Configuration (`config.toml`)

The evaluation uses two LLM configurations: one for the coding agent and one for the simulated user (the "fake user"). Add both to `evaluation/benchmarks/swe_bench/config.toml`:

```toml
# Agent backbone (Claude Sonnet 4.5 in the paper)
[llm.eval_claude_sonnet]
model = "anthropic/claude-sonnet-4-5-20250929"
api_key = "<your-anthropic-api-key>"

# Simulated user (GPT-5.1 in the paper)
[llm.fake_user]
model = "openai/gpt-5.1-2025-11-13"
api_key = "<your-openai-api-key>"
```

The `[llm.fake_user]` config is read by `hidden_run_infer.py`, `interact_run_infer.py`, `clarify_interact_run_infer.py`, and `clarify_v2_interact_run_infer.py` to instantiate the `FakeUser` simulator. The simulator is provided with the original fully-specified issue and constrained to answer queries from the coding agent using only that withheld context.

To run only a specific subset of SWE-bench instances, add a `selected_ids` key:

```toml
# Run only these specific instances (used for the 5 batches of 100 in the paper)
selected_ids = [
  "astropy__astropy-12907",
  "django__django-10097",
  # ...
]
```

The batch ID lists used in the paper are in `evaluation/benchmarks/swe_bench/scripts/docker/filtered_verified_images_batch{1..5}.txt`.

## 🚀 Running Experiments

The paper evaluates all five settings on **500 instances** (split into five batches of 100). Run each setting once per batch by swapping in the appropriate `selected_ids` list each time. Collect the resulting `output.jsonl` and evaluated `report.json` files under `analysis/final_evaluation_outputs/batch_{1..5}/{setting}/` for use in the analysis step.

Each evaluation setting has a dedicated inference script. All scripts share the same argument signature:

```bash
bash ./evaluation/benchmarks/swe_bench/scripts/<script>.sh \
  <model_config> <commit_hash> <agent> <eval_limit> <max_iter> <num_workers> <dataset> <split> <n_runs>
```

| Setting | Script | Default agent | Task prompt |
|---|---|---|---|
| Full | `run_infer.sh` | `CodeActAgent` | `swe_default.j2` — standard SWE-bench prompt with fully specified issue |
| Hidden | `hidden_run_infer.sh` | `CodeActAgent` | `swe_default.j2` — standard SWE-bench prompt with underspecified issue |
| Interactive Baseline | `interact_run_infer.sh` | `CodeActAgent` | `swe_default_interact.j2` — adds explicit instruction to ask a question before proceeding |
| UA-Single | `clarify_v2_interact_run_infer.sh` | `ClarifyAgentV2` | `swe_default.j2` — standard SWE-bench prompt with underspecified issue |
| UA-Multi | `clarify_interact_run_infer.sh` | `ClarifyAgent` | `swe_default.j2` — standard SWE-bench prompt with underspecified issue |

### Example commands

```bash
# Full baseline (fully specified issues, no interaction)
bash ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
  llm.eval_claude_sonnet HEAD CodeActAgent 100 100 1 \
  princeton-nlp/SWE-bench_Verified test

# Hidden baseline (underspecified issues, no interaction)
bash ./evaluation/benchmarks/swe_bench/scripts/hidden_run_infer.sh \
  llm.eval_claude_sonnet HEAD CodeActAgent 100 100 1 \
  cmu-lti/interactive-swe test

# Interactive Baseline (underspecified + hardcoded interaction)
bash ./evaluation/benchmarks/swe_bench/scripts/interact_run_infer.sh \
  llm.eval_claude_sonnet HEAD CodeActAgent 100 100 1 \
  cmu-lti/interactive-swe test

# UA-Single (single-agent clarification)
bash ./evaluation/benchmarks/swe_bench/scripts/clarify_v2_interact_run_infer.sh \
  llm.eval_claude_sonnet HEAD ClarifyAgentV2 100 100 1 \
  cmu-lti/interactive-swe test

# UA-Multi (multi-agent clarification)
# NOTE: max_iter is set to 300 (not 100) because each coding agent turn triggers a
# delegate call to the Intent Agent, which counts as an additional 2 turns. 300 iterations
# is equivalent to ~100 effective coding-agent turns, matching the other settings.
bash ./evaluation/benchmarks/swe_bench/scripts/clarify_interact_run_infer.sh \
  llm.eval_claude_sonnet HEAD ClarifyAgent 100 300 1 \
  cmu-lti/interactive-swe test
```

### 📦 Dataset

The dataset [`cmu-lti/interactive-swe`](https://huggingface.co/datasets/cmu-lti/interactive-swe) is the underspecified variant of SWE-bench Verified introduced by [Vijayvargiya et al. (2026)](https://arxiv.org/abs/2502.13069) ([codebase](https://github.com/sani903/InteractiveSWEAgents)), used for all interactive settings. `princeton-nlp/SWE-bench_Verified` is used for the Full baseline.

## 🏗️ Agent Architectures

### UA-Multi: `ClarifyAgent` + `IntentAgent`
Implemented in [openhands/agenthub/clarify_agent/](openhands/agenthub/clarify_agent/) and [openhands/agenthub/intent_agent/](openhands/agenthub/intent_agent/). The **Main Agent** (`ClarifyAgent`) handles all code execution. After each main-agent turn, the **Intent Agent** analyses the conversation history and calls `clarify_decision` to signal whether clarification is required. If it signals `needs_clarification: true`, the Main Agent's next action is constrained to issue a `clarify` call to the simulated user.

### UA-Single: `ClarifyAgentV2`
Implemented in [openhands/agenthub/clarify_agent_v2/](openhands/agenthub/clarify_agent_v2/). A single agent is prompted at each turn with a reminder to assess for underspecification. If it detects missing information, it calls the `clarify` tool directly.

### Interactive Baseline: `CodeActAgent` with `swe_default_interact.j2`
A standard `CodeActAgent` run with a modified task prompt ([evaluation/benchmarks/swe_bench/prompts/swe_default_interact.j2](evaluation/benchmarks/swe_bench/prompts/swe_default_interact.j2)) that explicitly instructs the agent to ask questions before proceeding.

### Full / Hidden: `CodeActAgent` with `swe_default.j2`
Standard `CodeActAgent` receiving the fully specified (Full) or underspecified (Hidden) issue using the unmodified SWE-bench task prompt ([evaluation/benchmarks/swe_bench/prompts/swe_default.j2](evaluation/benchmarks/swe_bench/prompts/swe_default.j2)).

## ✅ Evaluating Results

After inference, evaluate generated patches with the official SWE-bench harness:

```bash
./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
  <path/to/output.jsonl> "" princeton-nlp/SWE-bench_Verified test
```

See [evaluation/benchmarks/swe_bench/README.md](evaluation/benchmarks/swe_bench/README.md) for full evaluation documentation.

## ⚠️ Known Evaluation Issues

**SWE-bench patch revert bug (sphinx / astropy instances)**: A bug in the SWE-bench evaluation harness package (`swebench`) causes applied patches to be silently reverted to the base commit state for certain `sphinx` and `astropy` instances, causing all those runs to fail regardless of patch quality. This results in roughly 10% lower reported resolve rates compared to the official OpenHands numbers if left unpatched. Apply the fix from [Kipok/SWE-bench#3](https://github.com/Kipok/SWE-bench/pull/3) (tracked in [SWE-bench/SWE-bench#228](https://github.com/SWE-bench/SWE-bench/issues/228)) to resolve this. The paper results were produced with this fix applied.

## 🔬 Analysis and Paper Reproduction

The `analysis/` directory contains scripts to reproduce all tables and figures from the paper. See [analysis/reproduction.md](analysis/reproduction.md) for the full reproduction guide.

`analysis/original_paper_results/` contains the exact tables and figures as they appear in the submitted paper for reference.

The analysis scripts expect evaluation outputs and trajectories collected across all five batches under `analysis/final_evaluation_outputs/` and `analysis/final_trajectories/` respectively (see [analysis/reproduction.md](analysis/reproduction.md) for the full directory layout).

### Running the analysis

The recommended end-to-end command, run from the `analysis/` directory:

```bash
# Step 1 + 2 combined: build QA artifacts from trajectories, then produce paper outputs.
# Requires ANTHROPIC_API_KEY for token counting.
export ANTHROPIC_API_KEY=...
python3 scripts/prepare_paper_rebuild_inputs.py \
  --rebuild-after \
  --rebuild-out-dir paper_reproduction_outputs/from_raw
```

`prepare_paper_rebuild_inputs.py` handles both steps: it first builds the intermediate QA pair artifacts under `qa_pairs/` from the raw trajectories, then calls `rebuild_paper_results_from_raw.py` to produce the final tables and figures. If the QA artifacts already exist, you can skip straight to step 2:

```bash
# Step 2 only: reproduce tables and figures from pre-built QA artifacts.
# Does not require ANTHROPIC_API_KEY.
python3 scripts/rebuild_paper_results_from_raw.py \
  --out-dir paper_reproduction_outputs/from_raw
```

Outputs are written to `analysis/paper_reproduction_outputs/from_raw/` by default (configurable via `--out-dir` / `--rebuild-out-dir`).

Key scripts:

| Script | Purpose |
|---|---|
| `analysis/scripts/prepare_paper_rebuild_inputs.py` | Builds QA pair artifacts from raw trajectories; optionally calls `rebuild_paper_results_from_raw.py` via `--rebuild-after` |
| `analysis/scripts/rebuild_paper_results_from_raw.py` | Produces paper tables and figures from QA artifacts + eval outputs |
| `analysis/scripts/compute_ask_significance.py` | Computes significance statistics for ask-rate comparisons (optional) |
| `analysis/qa_scripts/qa_cli.py` | Low-level QA pipeline called internally by `prepare_paper_rebuild_inputs.py` |

## 🙏 Acknowledgements

This work builds on the [OpenHands](https://github.com/All-Hands-AI/OpenHands) agent framework developed by All-Hands-AI. We use the interactive SWE-bench dataset introduced by [Vijayvargiya et al. (2026)](https://arxiv.org/abs/2502.13069) ([dataset](https://huggingface.co/datasets/cmu-lti/interactive-swe), [codebase](https://github.com/sani903/InteractiveSWEAgents)).

## 📖 Citation

If you use this work, please consider citing our paper:

```bibtex
@misc{edwards2026askassumeuncertaintyawareclarificationseeking,
      title={Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents}, 
      author={Nicholas Edwards and Sebastian Schuster},
      year={2026},
      eprint={2603.26233},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.26233}, 
}
```
