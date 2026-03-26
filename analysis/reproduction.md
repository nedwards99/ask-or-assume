# Analysis Reproduction

This document describes how to reproduce the paper results from raw evaluation
artifacts. All scripts are designed to be run from the `analysis/` directory.

---

## Background

This analysis evaluates whether allowing a coding agent to ask clarifying
questions about underspecified SWE-bench task descriptions improves its ability to
resolve issues. The study uses the **SWE-bench Verified** dataset, split into five
batches of 100 tasks each. Each batch is run under multiple agent
**settings** (described below), and the resulting trajectories and evaluation
outputs are collected here for analysis.

### Agent Settings

| Setting | Short name | Description |
|---|---|---|
| `clarify_interact` | UA-Multi | Multi-agent clarification system. A clarify sub-agent decides whether to ask the user a question before the coding agent proceeds. |
| `clarify_interact_v2` | UA-Single | Single-agent clarification system. The coding agent itself decides when to ask the user a clarifying question. |
| `interact` | Interactive Baseline | Standard interactive agent — explicitly prompted to ask questions before proceeding. |
| `full` | Full | Standard agent with the fully specified original GitHub issue. No user interaction. Upper-bound baseline. |
| `hidden` | Hidden | Standard agent with the underspecified issue. No user interaction. Lower-bound baseline. |

The central research question is whether agents in the `clarify_interact` and
`clarify_interact_v2` settings ask questions on harder/more ambiguous tasks,
and whether asking improves their resolve rate on those tasks.

### Task Difficulty Annotations

`swe-bench-annotation-results/ensembled_annotations_public.csv` contains
human-annotated labels from the original [SWE-Bench Verified benchmark](https://openai.com/index/introducing-swe-bench-verified/) for each of the 500 instances, used to compute the ask rates by task difficulty.

---

## Data Layout

The scripts assume this directory structure under `analysis/`. Files and
directories marked **[provided]** are included in this repository. Directories
marked **[not provided]** must be populated by running the evaluation (see the
main README) or by obtaining the raw outputs separately.

```text
analysis/
  reproduction.md                          ← this file [provided]
  scripts/                                 ← [provided]
    prepare_paper_rebuild_inputs.py        ← step 1: build qa_pairs/ from trajectories
    rebuild_paper_results_from_raw.py      ← step 2: build tables/figures from qa_pairs/ + eval outputs
    compute_ask_significance.py            ← optional significance stats
  qa_scripts/                              ← [provided]
    qa_cli.py                              ← low-level QA pipeline; called by prepare_paper_rebuild_inputs.py
  swe-bench-annotation-results/            ← [provided]
    ensembled_annotations_public.csv       ← per-instance difficulty/underspecification labels
  original_paper_results/                  ← [provided] exact tables and figures from the submitted paper
  final_trajectories/                      ← [not provided] raw agent trajectories from inference runs
    clarify_interact/batch_{1..5}/...
    clarify_interact_v2/batch_{1..5}/...
    interact/batch_{1..5}/...
  final_evaluation_outputs/                ← [not provided] SWE-bench evaluation results from eval runs
    batch_{1..5}/
      {clarify_interact,clarify_interact_v2,interact,full,hidden}/
        report.json                        ← aggregate resolve-rate stats
        output.jsonl                       ← per-instance results + cost data
  qa_pairs/                                ← [not provided initially] built by prepare_paper_rebuild_inputs.py
    batch_{1..5}/
      {clarify_interact,clarify_interact_v2,interact_with_qmark}/
        *_qa_pairs.jsonl
        *_qa_pairs_positions.jsonl
        reports/
          *_token_report.jsonl
          *_decile_token_report.json
          *_bucket_token_report.json
```

Note: `interact_with_qmark` is not a separate evaluation run. It is derived
from the `interact` trajectories by filtering to only assistant messages
containing `?`, used to compare question-asking behaviour against the UA agents.

`qa_pairs/` is created by `prepare_paper_rebuild_inputs.py` and does not need
to exist before running that script.

---

## Batches

The 500 instances are split into five non-overlapping batches of 100 tasks each,
so that inference can be run in parallel across machines. Batch membership is
defined in `final_batches/batch{1..5}.toml` (TOML array of SWE-bench instance
IDs) and mirrored in `final_batches/batch{1..5}_ids.txt` (plain text, one ID
per line).

The analysis scripts expect results organised by batch, as shown in the layout
above.

---

## Trajectory Files

`final_trajectories/` holds the raw agent trajectories — one JSON file per
(instance, run) pair, named `{instance_id}_{timestamp}.json`.

Each file records the full sequence of agent steps: tool calls, observations,
and (in the clarify settings) any clarifying questions the agent chose to ask
and the simulated user responses it received.

Trajectories are organised as:

```text
final_trajectories/{setting}/batch_{n}/{model_dir}/  ← one nested model directory
```

or, for some batches, with JSON files directly under `batch_{n}/`.
`prepare_paper_rebuild_inputs.py` handles both layouts automatically.

---

## Evaluation Outputs

`final_evaluation_outputs/batch_{n}/{setting}/` contains the SWE-bench
evaluation results for one batch under one setting:

- **`report.json`** — aggregate counts: total instances, submitted, resolved,
  unresolved, empty-patch, error. Primary source for resolve rates.
- **`output.jsonl`** — one record per instance with the generated patch, test
  results, and token/cost data used for the cost table.

---

## QA Pairs

`qa_pairs/` holds intermediate artifacts derived from trajectories that
characterise the clarifying questions the agent asked.

For each batch × setting combination, `qa_cli.py` produces:

| File | What it contains |
|---|---|
| `*_qa_pairs.jsonl` | One record per question–answer pair extracted from a trajectory: the question text, the answer text, and the instance ID. |
| `*_qa_pairs_positions.jsonl` | The same pairs enriched with positional metadata: where in the trajectory the question appeared (as a fraction of total events and by event index), used to analyse *when* agents ask questions. |
| `reports/*_token_report.jsonl` | Per-pair token counts for the question and answer, as reported by Anthropic's token counting API. |
| `reports/*_decile_token_report.json` | Token-count distribution summarised by decile, used to characterise question length. |
| `reports/*_bucket_token_report.json` | Token-count distribution summarised by fixed bucket (early/mid/late trajectory), used for the timing analysis in the paper. |

---

## Scripts

### `prepare_paper_rebuild_inputs.py`

Walks `final_trajectories/` for the three QA-relevant settings and generates
the `qa_pairs/` artifacts above. It is a wrapper around `qa_cli.py` that
handles the batch-and-setting loop, skips already-present files, and optionally
triggers the downstream paper rebuild via `--rebuild-after`.

**Requires `ANTHROPIC_API_KEY`** when token reports need to be computed
(`reports/*_token_report.jsonl` files are absent), because `qa_cli.py
token-count` calls Anthropic's token counting API.

### `rebuild_paper_results_from_raw.py`

Combines the `qa_pairs/` artifacts with `final_evaluation_outputs/` and
`swe-bench-annotation-results/ensembled_annotations_public.csv` to produce the
packaged paper outputs under `--out-dir`:

```text
{out-dir}/
  tables/          ← paper tables as CSV, Markdown, LaTeX
  figures/         ← resolve-rate figure as PNG + PDF
  intermediates/   ← per-batch summaries used to build resolve rate and cost tables
  rebuild_manifest.json
```

Does **not** require `ANTHROPIC_API_KEY` if all `qa_pairs/` artifacts already
exist.

### `compute_ask_significance.py`

Optional. Computes significance statistics for the ask-vs-difficulty and
ask-rate-difference comparisons reported in the paper. Not part of the main
end-to-end rebuild flow.

### `qa_scripts/qa_cli.py`

Low-level QA pipeline tool called internally by `prepare_paper_rebuild_inputs.py`.
You generally do not need to invoke it directly for paper reproduction; see its
`--help` output for details.

---

## Commands

### End-to-end rebuild (starting from trajectories)

```bash
export ANTHROPIC_API_KEY=...
python3 scripts/prepare_paper_rebuild_inputs.py \
  --rebuild-after \
  --rebuild-out-dir paper_reproduction_outputs/from_raw
```

This runs both steps in sequence:
1. Builds all `qa_pairs/` artifacts from trajectories (calls Anthropic token counting API)
2. Runs `rebuild_paper_results_from_raw.py` and writes final paper outputs

### Rebuild tables and figures only (QA artifacts already present)

```bash
python3 scripts/rebuild_paper_results_from_raw.py \
  --out-dir paper_reproduction_outputs/from_raw
```

Use this when `qa_pairs/` already contains all expected files. Does not require
`ANTHROPIC_API_KEY`.

---

## Credentials

`ANTHROPIC_API_KEY` is required only when `qa_cli.py token-count` must be run,
i.e. when `reports/*_token_report.jsonl` files are absent. If all token
artifacts already exist, both scripts can run without it.
