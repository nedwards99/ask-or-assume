#!/usr/bin/env python3
"""Rebuild paper results from batch-level artifacts.

This script is a higher-level reproducibility entrypoint than
`generate_paper_result_tables.py`. It starts from batch-level evaluation
artifacts under `final_evaluation_outputs/` and `qa_pairs/` and regenerates:

- combined overall resolve-rate summary
- combined cost summary
- paper tables (CSV + Markdown)
- optional figure files if matplotlib is installed
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
BATCH_DEFAULT = [1, 2, 3, 4, 5]
SETTINGS = ["full", "clarify_interact", "clarify_interact_v2", "interact", "hidden"]
SETTING_LABELS = {
    "clarify_interact": "Uncertainty-Aware (Multi)",
    "clarify_interact_v2": "Uncertainty-Aware (Single)",
    "full": "Full",
    "hidden": "Hidden",
    "interact": "Interactive Baseline",
}
PLOT_ORDER = ["full", "hidden", "interact", "clarify_interact_v2", "clarify_interact"]
PLOT_COLORS = {
    "full": "#4C78A8",
    "hidden": "#E45756",
    "interact": "#A8C6A1",
    "clarify_interact_v2": "#54A24B",
    "clarify_interact": "#2E7D32",
}
PLOT_FAMILIES = {
    "full": "Full",
    "hidden": "Hidden",
    "interact": "Interaction",
    "clarify_interact_v2": "Interaction",
    "clarify_interact": "Interaction",
}
SIGNIFICANCE_LABELS = {
    ("clarify_interact_v2", "clarify_interact"): "***",
    ("interact", "clarify_interact"): "n.s.",
    ("hidden", "clarify_interact"): "***",
    ("full", "clarify_interact"): "n.s.",
}
DIFFICULTY_ORDER = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]


def repo_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def parse_batches(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_markdown_table(title: str, columns: List[str], rows: List[dict]) -> str:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_csv_row(path: Path, **filters: str) -> dict:
    for row in read_csv_rows(path):
        if all(str(row.get(key)) == str(value) for key, value in filters.items()):
            return row
    raise KeyError(f"Missing row in {path}: {filters}")


def find_single(pattern: str) -> Path:
    paths = [Path(path) for path in glob.glob(pattern)]
    if len(paths) != 1:
        raise RuntimeError(f"Expected one path for {pattern!r}, found {len(paths)}")
    return paths[0]


def instance_id_from_trajectory(name: str) -> str:
    stem = name[:-5] if name.endswith(".json") else name
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def asked_ids_for_mode(qa_root: Path, batches: List[int], mode: str) -> Dict[int, set[str]]:
    out: Dict[int, set[str]] = {}
    for batch in batches:
        qa_path = find_single(str(qa_root / f"batch_{batch}" / mode / "*_qa_pairs.jsonl"))
        asked = set()
        for record in load_jsonl(qa_path):
            trajectory = record.get("trajectory") or ""
            if trajectory:
                asked.add(instance_id_from_trajectory(trajectory))
        out[batch] = asked
    return out


def aggregate_overall_reports(report_root: Path, batches: List[int]) -> List[dict]:
    overall = {setting: Counter() for setting in SETTINGS}
    for batch in batches:
        for setting in SETTINGS:
            report = load_json(report_root / f"batch_{batch}" / setting / "report.json")
            submitted = set(report.get("submitted_ids") or [])
            resolved = set(report.get("resolved_ids") or [])
            unresolved = submitted - resolved
            overall[setting]["submitted"] += len(submitted)
            overall[setting]["resolved"] += len(resolved)
            overall[setting]["unresolved"] += len(unresolved)

    rows = []
    for setting in SETTINGS:
        submitted = overall[setting]["submitted"]
        resolved = overall[setting]["resolved"]
        unresolved = overall[setting]["unresolved"]
        rows.append(
            {
                "setting": setting,
                "resolved": resolved,
                "unresolved": unresolved,
                "submitted": submitted,
                "resolve_rate_pct": round((resolved / submitted) * 100, 2) if submitted else 0.0,
            }
        )
    return rows


def aggregate_costs(report_root: Path, batches: List[int]) -> List[dict]:
    rows = {setting: defaultdict(float) for setting in SETTINGS}
    for batch in batches:
        for setting in SETTINGS:
            path = report_root / f"batch_{batch}" / setting / "output.jsonl"
            instances = 0
            total_cost = 0.0
            for record in load_jsonl(path):
                instances += 1
                metrics = record.get("metrics") or {}
                cost = metrics.get("accumulated_cost")
                if cost is None:
                    costs = metrics.get("costs") or []
                    cost = sum(entry.get("cost", 0.0) for entry in costs if isinstance(entry, dict))
                total_cost += float(cost or 0.0)
            rows[setting]["instances"] += instances
            rows[setting]["total_cost"] += total_cost

    out = []
    for setting in SETTINGS:
        instances = int(rows[setting]["instances"])
        total = float(rows[setting]["total_cost"])
        out.append(
            {
                "setting": setting,
                "instances": instances,
                "total_cost": round(total, 6),
                "avg_cost_per_instance": round((total / instances), 6) if instances else 0.0,
            }
        )
    return out


def summarize_cross_subset(qa_root: Path, report_root: Path, batches: List[int], subset_mode: str, setting: str) -> dict:
    asked_by_batch = asked_ids_for_mode(qa_root, batches, subset_mode)
    population_total = 0
    asked_total = 0
    asked_resolved = 0
    not_asked_total = 0
    not_asked_resolved = 0

    for batch in batches:
        report = load_json(report_root / f"batch_{batch}" / setting / "report.json")
        population = set(report.get("submitted_ids") or [])
        resolved = set(report.get("resolved_ids") or [])
        asked = population & asked_by_batch[batch]
        not_asked = population - asked
        population_total += len(population)
        asked_total += len(asked)
        asked_resolved += len(asked & resolved)
        not_asked_total += len(not_asked)
        not_asked_resolved += len(not_asked & resolved)

    return {
        "population_total": population_total,
        "asked_total": asked_total,
        "asked_resolved": asked_resolved,
        "asked_rate_pct": round((asked_resolved / asked_total) * 100, 2) if asked_total else 0.0,
        "not_asked_total": not_asked_total,
        "not_asked_resolved": not_asked_resolved,
        "not_asked_rate_pct": round((not_asked_resolved / not_asked_total) * 100, 2) if not_asked_total else 0.0,
    }


def aggregate_token_file_summary(qa_root: Path, batches: List[int], mode: str) -> dict:
    totals = {
        "trajectories": 0,
        "questions_total": 0,
        "question_tokens_total": 0,
        "answer_tokens_total": 0,
    }
    for batch in batches:
        token_path = find_single(str(qa_root / f"batch_{batch}" / mode / "reports" / "*_token_report.jsonl"))
        first = next(load_jsonl(token_path))
        totals["trajectories"] += int(first.get("trajectories", 0))
        totals["questions_total"] += int(first.get("questions_total", 0))
        totals["question_tokens_total"] += int(first.get("question_tokens_total", 0))
        totals["answer_tokens_total"] += int(first.get("answer_tokens_total", 0))
    return totals


def aggregate_trajectory_question_counts(qa_root: Path, batches: List[int], mode: str) -> dict:
    trajectories = 0
    questions_total = 0
    for batch in batches:
        token_path = find_single(str(qa_root / f"batch_{batch}" / mode / "reports" / "*_token_report.jsonl"))
        for row in load_jsonl(token_path):
            if row.get("type") != "trajectory":
                continue
            trajectories += 1
            questions_total += int(row.get("num_questions", 0))
    return {"trajectories": trajectories, "questions_total": questions_total}


def aggregate_bucket_counts(qa_root: Path, batches: List[int], mode: str) -> Dict[str, int]:
    counts = {"early": 0, "mid": 0, "late": 0}
    for batch in batches:
        bucket_path = find_single(str(qa_root / f"batch_{batch}" / mode / "reports" / "*_bucket_token_report.json"))
        for row in load_json(bucket_path):
            if str(row.get("category", "all")) != "all":
                continue
            counts[str(row["bucket"])] += int(row.get("questions", 0))
    return counts


def load_difficulty_map(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            iid = row.get("instance_id", "").strip()
            diff = row.get("difficulty", "").strip()
            if iid and diff:
                out[iid] = diff
    return out


def build_difficulty_rows(qa_root: Path, report_root: Path, batches: List[int], mode: str, difficulty_map_csv: Path) -> List[dict]:
    asked_by_batch = asked_ids_for_mode(qa_root, batches, mode)
    difficulty_by_id = load_difficulty_map(difficulty_map_csv)
    counts = {difficulty: {"total": 0, "asked": 0} for difficulty in DIFFICULTY_ORDER}

    for batch in batches:
        report = load_json(report_root / f"batch_{batch}" / mode / "report.json")
        submitted = set(report.get("submitted_ids") or [])
        asked = asked_by_batch[batch]
        for instance_id in submitted:
            difficulty = difficulty_by_id.get(instance_id)
            if difficulty not in counts:
                continue
            counts[difficulty]["total"] += 1
            counts[difficulty]["asked"] += int(instance_id in asked)

    rows = []
    for difficulty in DIFFICULTY_ORDER:
        total = counts[difficulty]["total"]
        asked = counts[difficulty]["asked"]
        rows.append(
            {
                "difficulty_level": difficulty,
                "tasks": total,
                "ask_rate_pct": round((asked / total) * 100, 2) if total else 0.0,
            }
        )
    return rows


def build_question_stats_rows(
    qa_root: Path,
    batches: List[int],
) -> tuple[List[dict], dict]:
    multi_totals = aggregate_token_file_summary(qa_root, batches, "clarify_interact")
    multi_buckets = aggregate_bucket_counts(qa_root, batches, "clarify_interact")

    v2_trajectory = aggregate_trajectory_question_counts(qa_root, batches, "clarify_interact_v2")
    v2_totals_raw = aggregate_token_file_summary(qa_root, batches, "clarify_interact_v2")
    v2_buckets = aggregate_bucket_counts(qa_root, batches, "clarify_interact_v2")

    baseline_totals = aggregate_token_file_summary(qa_root, batches, "interact_with_qmark")
    baseline_trajectory = aggregate_trajectory_question_counts(qa_root, batches, "interact_with_qmark")
    baseline_buckets = aggregate_bucket_counts(qa_root, batches, "interact_with_qmark")

    v2_total_queries = v2_trajectory["questions_total"]
    v2_q_avg = round(v2_totals_raw["question_tokens_total"] / v2_total_queries, 2) if v2_total_queries else 0.0
    v2_a_avg = round(v2_totals_raw["answer_tokens_total"] / v2_total_queries, 2) if v2_total_queries else 0.0

    def dist_pct(counts: Dict[str, int]) -> dict:
        total = sum(counts.values())
        return {
            "early_pct": round((counts["early"] / total) * 100, 1) if total else 0.0,
            "mid_pct": round((counts["mid"] / total) * 100, 1) if total else 0.0,
            "late_pct": round((counts["late"] / total) * 100, 1) if total else 0.0,
        }

    rows = [
        {
            "agent": "Uncertainty-Aware (Multi)",
            "queried_tasks": multi_totals["trajectories"],
            "total_queries": multi_totals["questions_total"],
            "avg_queries_per_task": round(multi_totals["questions_total"] / multi_totals["trajectories"], 2),
            "avg_question_tokens": round(multi_totals["question_tokens_total"] / multi_totals["questions_total"], 2),
            "avg_answer_tokens": round(multi_totals["answer_tokens_total"] / multi_totals["questions_total"], 2),
            **dist_pct(multi_buckets),
        },
        {
            "agent": "Uncertainty-Aware (Single)",
            "queried_tasks": v2_trajectory["trajectories"],
            "total_queries": v2_total_queries,
            "avg_queries_per_task": round(v2_total_queries / v2_trajectory["trajectories"], 2),
            "avg_question_tokens": v2_q_avg,
            "avg_answer_tokens": v2_a_avg,
            **dist_pct(v2_buckets),
        },
        {
            "agent": "Interactive Baseline",
            "queried_tasks": baseline_trajectory["trajectories"],
            "total_queries": baseline_trajectory["questions_total"],
            "avg_queries_per_task": round(baseline_trajectory["questions_total"] / baseline_trajectory["trajectories"], 2),
            "avg_question_tokens": round(baseline_totals["question_tokens_total"] / baseline_totals["questions_total"], 2),
            "avg_answer_tokens": round(baseline_totals["answer_tokens_total"] / baseline_totals["questions_total"], 2),
            **dist_pct(baseline_buckets),
        },
    ]
    metadata = {
        "raw_v2_questions_total": v2_trajectory["questions_total"],
        "reported_v2_questions_total": v2_total_queries,
    }
    return rows, metadata


def write_figure_from_rows(rows: List[dict], output_png: Path, output_pdf: Path, output_json: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    by_setting = {row["setting"]: row for row in rows}
    settings = [setting for setting in PLOT_ORDER if setting in by_setting]
    rates = [float(by_setting[setting]["resolve_rate_pct"]) for setting in settings]
    labels = [SETTING_LABELS[setting].replace(" (", "\n(") if setting.startswith("clarify") else SETTING_LABELS[setting].replace(" ", "\n", 1) if setting == "interact" else SETTING_LABELS[setting] for setting in settings]
    colors = [PLOT_COLORS[setting] for setting in settings]
    families = [PLOT_FAMILIES[setting] for setting in settings]

    x_positions = []
    current_x = 0.0
    previous_family = None
    for family in families:
        if previous_family is not None and family != previous_family:
            current_x += 0.28
        x_positions.append(current_x)
        current_x += 1.0
        previous_family = family

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(x_positions, rates, color=colors, edgecolor="#222222", linewidth=0.8, width=0.68)
    ax.set_ylabel("Resolve Rate (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for idx in range(len(settings) - 1):
        if families[idx] != families[idx + 1]:
            divider_x = (x_positions[idx] + x_positions[idx + 1]) / 2
            ax.axvline(divider_x, color="#BBBBBB", linewidth=1.0, linestyle=":")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{rate:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    index_by_setting = {setting: idx for idx, setting in enumerate(settings)}
    bracket_y = max(rates) + 4.6
    bracket_gap = 5.0
    bracket_height = 1.1
    bracket_specs = []
    for order, ((left, right), label) in enumerate(SIGNIFICANCE_LABELS.items()):
        if left not in index_by_setting or right not in index_by_setting:
            continue
        left_idx = index_by_setting[left]
        right_idx = index_by_setting[right]
        x1 = x_positions[min(left_idx, right_idx)]
        x2 = x_positions[max(left_idx, right_idx)]
        bracket_specs.append({"order": order, "left_setting": left, "right_setting": right, "label": label, "left_x": x1, "right_x": x2})
        y = bracket_y + (len(bracket_specs) - 1) * bracket_gap
        ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color="#000000", linewidth=1.0)
        ax.text((x1 + x2) / 2, y + bracket_height + 0.4, label, ha="center", va="bottom", fontsize=8, color="#000000")

    ax.set_ylim(0, max(rates) + 4.6 + max(1, len(bracket_specs)) * bracket_gap + 3.0)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.18, left=0.10, right=0.99, top=0.98)
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf, dpi=180)
    plt.close(fig)

    write_json(
        output_json,
        {
            "settings": settings,
            "rates": rates,
            "significance_brackets": bracket_specs,
        },
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild paper results from batch-level raw artifacts.")
    parser.add_argument("--report-root", default="final_evaluation_outputs")
    parser.add_argument("--qa-root", default="qa_pairs")
    parser.add_argument("--difficulty-map-csv", default="swe-bench-annotation-results/ensembled_annotations_public.csv")
    parser.add_argument("--batches", default="1,2,3,4,5")
    parser.add_argument("--out-dir", default="paper_reproduction_outputs/from_raw")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    report_root = repo_path(args.report_root)
    qa_root = repo_path(args.qa_root)
    difficulty_csv = repo_path(args.difficulty_map_csv)
    out_dir = repo_path(args.out_dir)
    tables_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    intermediates_dir = out_dir / "intermediates"
    batches = parse_batches(args.batches)

    overall_rows = aggregate_overall_reports(report_root, batches)
    cost_rows_raw = aggregate_costs(report_root, batches)

    write_csv(intermediates_dir / "combined_overall_by_setting.csv", overall_rows, ["setting", "resolved", "unresolved", "submitted", "resolve_rate_pct"])
    write_csv(intermediates_dir / "combined_cost_by_setting.csv", cost_rows_raw, ["setting", "instances", "total_cost", "avg_cost_per_instance"])

    multi_rows = []
    for setting in ["clarify_interact", "full", "hidden", "clarify_interact_v2", "interact"]:
        stat = summarize_cross_subset(qa_root, report_root, batches, "clarify_interact", setting)
        multi_rows.append(
            {
                "evaluation_setting": SETTING_LABELS[setting],
                "asked_resolved": stat["asked_resolved"],
                "asked_rate_pct": stat["asked_rate_pct"],
                "not_asked_resolved": stat["not_asked_resolved"],
                "not_asked_rate_pct": stat["not_asked_rate_pct"],
            }
        )

    single_rows = []
    for setting in ["clarify_interact_v2", "full", "hidden", "clarify_interact", "interact"]:
        stat = summarize_cross_subset(qa_root, report_root, batches, "clarify_interact_v2", setting)
        single_rows.append(
            {
                "evaluation_setting": SETTING_LABELS[setting],
                "asked_resolved": stat["asked_resolved"],
                "asked_rate_pct": stat["asked_rate_pct"],
                "not_asked_resolved": stat["not_asked_resolved"],
                "not_asked_rate_pct": stat["not_asked_rate_pct"],
            }
        )

    single_difficulty = build_difficulty_rows(qa_root, report_root, batches, "clarify_interact_v2", difficulty_csv)
    multi_difficulty = build_difficulty_rows(qa_root, report_root, batches, "clarify_interact", difficulty_csv)
    multi_diff_by_level = {row["difficulty_level"]: row for row in multi_difficulty}
    difficulty_rows = [
        {
            "difficulty_level": row["difficulty_level"],
            "tasks": row["tasks"],
            "ua_single_ask_rate_pct": row["ask_rate_pct"],
            "ua_multi_ask_rate_pct": multi_diff_by_level[row["difficulty_level"]]["ask_rate_pct"],
        }
        for row in single_difficulty
    ]

    question_rows, question_meta = build_question_stats_rows(qa_root, batches)
    cost_rows = [
        {
            "setting": SETTING_LABELS[row["setting"]],
            "total_cost_usd": round(float(row["total_cost"]), 2),
            "avg_cost_per_task_usd": round(float(row["avg_cost_per_instance"]), 2),
        }
        for row in sorted(cost_rows_raw, key=lambda row: ["full", "hidden", "interact", "clarify_interact_v2", "clarify_interact"].index(row["setting"]))
    ]

    write_csv(tables_dir / "ask_resolve_rate_multi.csv", multi_rows, ["evaluation_setting", "asked_resolved", "asked_rate_pct", "not_asked_resolved", "not_asked_rate_pct"])
    write_csv(tables_dir / "ask_resolve_rate_single.csv", single_rows, ["evaluation_setting", "asked_resolved", "asked_rate_pct", "not_asked_resolved", "not_asked_rate_pct"])
    write_csv(tables_dir / "difficulty_ask_rates.csv", difficulty_rows, ["difficulty_level", "tasks", "ua_single_ask_rate_pct", "ua_multi_ask_rate_pct"])
    write_csv(tables_dir / "question_stats.csv", question_rows, ["agent", "queried_tasks", "total_queries", "avg_queries_per_task", "avg_question_tokens", "avg_answer_tokens", "early_pct", "mid_pct", "late_pct"])
    write_csv(tables_dir / "combined_cost.csv", cost_rows, ["setting", "total_cost_usd", "avg_cost_per_task_usd"])

    write_text(tables_dir / "ask_resolve_rate_multi.md", render_markdown_table("Ask Resolve Rate (Multi)", ["evaluation_setting", "asked_resolved", "asked_rate_pct", "not_asked_resolved", "not_asked_rate_pct"], multi_rows))
    write_text(tables_dir / "ask_resolve_rate_single.md", render_markdown_table("Ask Resolve Rate (Single)", ["evaluation_setting", "asked_resolved", "asked_rate_pct", "not_asked_resolved", "not_asked_rate_pct"], single_rows))
    write_text(tables_dir / "difficulty_ask_rates.md", render_markdown_table("Difficulty Ask Rates", ["difficulty_level", "tasks", "ua_single_ask_rate_pct", "ua_multi_ask_rate_pct"], difficulty_rows))
    write_text(tables_dir / "question_stats.md", render_markdown_table("Question Stats", ["agent", "queried_tasks", "total_queries", "avg_queries_per_task", "avg_question_tokens", "avg_answer_tokens", "early_pct", "mid_pct", "late_pct"], question_rows))
    write_text(tables_dir / "combined_cost.md", render_markdown_table("Combined Cost", ["setting", "total_cost_usd", "avg_cost_per_task_usd"], cost_rows))

    figure_generated = False
    if not args.skip_figure:
        figure_generated = write_figure_from_rows(
            overall_rows,
            figure_dir / "combined_overall_by_setting_resolve_rates_pvalues_final_large.png",
            figure_dir / "combined_overall_by_setting_resolve_rates_pvalues_final_large.pdf",
            figure_dir / "combined_overall_by_setting_resolve_rates_pvalues_final_large.json",
        )

    manifest = {
        "batches": batches,
        "inputs": {
            "report_root": str(report_root.relative_to(ROOT)),
            "qa_root": str(qa_root.relative_to(ROOT)),
            "difficulty_map_csv": str(difficulty_csv.relative_to(ROOT)),
        },
        "required_batch_level_files": [
            "final_evaluation_outputs/batch_{b}/{full,hidden,interact,clarify_interact,clarify_interact_v2}/report.json",
            "final_evaluation_outputs/batch_{b}/{full,hidden,interact,clarify_interact,clarify_interact_v2}/output.jsonl",
            "qa_pairs/batch_{b}/{clarify_interact,clarify_interact_v2,interact_with_qmark}/*_qa_pairs.jsonl",
            "qa_pairs/batch_{b}/{clarify_interact,clarify_interact_v2,interact_with_qmark}/reports/*_token_report.jsonl",
            "qa_pairs/batch_{b}/{clarify_interact,clarify_interact_v2,interact_with_qmark}/reports/*_bucket_token_report.json",
        ],
        "question_stats_metadata": question_meta,
        "figure_generated": figure_generated,
        "figure_skipped_reason": None if figure_generated or args.skip_figure else "matplotlib_not_installed",
    }
    write_json(out_dir / "rebuild_manifest.json", manifest)

    print(tables_dir / "ask_resolve_rate_multi.md")
    print(tables_dir / "ask_resolve_rate_single.md")
    print(tables_dir / "difficulty_ask_rates.md")
    print(tables_dir / "question_stats.md")
    print(tables_dir / "combined_cost.md")
    print(intermediates_dir / "combined_overall_by_setting.csv")
    print(intermediates_dir / "combined_cost_by_setting.csv")
    print(out_dir / "rebuild_manifest.json")
    if figure_generated:
        print(figure_dir / "combined_overall_by_setting_resolve_rates_pvalues_final_large.png")
        print(figure_dir / "combined_overall_by_setting_resolve_rates_pvalues_final_large.pdf")
        print(figure_dir / "combined_overall_by_setting_resolve_rates_pvalues_final_large.json")
    elif args.skip_figure:
        print("Skipped figure generation (--skip-figure).")
    else:
        print("Skipped figure generation because matplotlib is not installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
