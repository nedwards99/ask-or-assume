#!/usr/bin/env python3
"""Prepare batch-level QA artifacts needed by rebuild_paper_results_from_raw.py.

This script fills in the batch/mode QA artifacts under `qa_pairs/` that the raw
paper rebuild expects:

- `*_qa_pairs.jsonl`
- `*_qa_pairs_positions.jsonl`
- `reports/*_token_report.jsonl`
- `reports/*_decile_token_report.json`
- `reports/*_bucket_token_report.json`

It reuses `qa_scripts/qa_cli.py` and only requires `ANTHROPIC_API_KEY` when it
needs to run `token-count`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModeConfig:
    output_mode: str
    trajectory_root: str
    qa_mode: str


MODE_CONFIGS = {
    "clarify_interact": ModeConfig(
        output_mode="clarify_interact",
        trajectory_root="final_trajectories/clarify_interact",
        qa_mode="clarify",
    ),
    "clarify_interact_v2": ModeConfig(
        output_mode="clarify_interact_v2",
        trajectory_root="final_trajectories/clarify_interact_v2",
        qa_mode="clarify",
    ),
    "interact_with_qmark": ModeConfig(
        output_mode="interact_with_qmark",
        trajectory_root="final_trajectories/interact",
        qa_mode="interact",
    ),
}


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [mode for mode in modes if mode not in MODE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown mode(s): {', '.join(unknown)}")
    return modes


def has_json_files(path: Path) -> bool:
    return any(path.glob("*.json"))


def discover_trajectory_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Trajectory batch directory not found: {base_dir}")
    if has_json_files(base_dir):
        return base_dir

    candidates = sorted(
        child for child in base_dir.iterdir() if child.is_dir() and has_json_files(child)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No trajectory JSON files found under {base_dir}")
    names = ", ".join(child.name for child in candidates)
    raise RuntimeError(f"Expected one trajectory directory under {base_dir}, found: {names}")


def ensure_single(path_glob: Iterable[Path], label: str, parent: Path) -> Path | None:
    matches = sorted(path_glob)
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise RuntimeError(f"Expected one {label} in {parent}, found: {names}")
    return matches[0]


def run_command(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(printable)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=ROOT)


def needs_token_api(will_run_token_count: bool) -> None:
    if will_run_token_count and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is required to build token reports. "
            "Set it in the environment or rerun with --skip-tokens."
        )


def prepare_mode_batch(
    batch: int,
    mode: str,
    config: ModeConfig,
    *,
    model: str,
    force: bool,
    skip_tokens: bool,
    dry_run: bool,
) -> None:
    batch_name = f"batch_{batch}"
    trajectory_dir = discover_trajectory_dir(ROOT / config.trajectory_root / batch_name)
    output_dir = ROOT / "qa_pairs" / batch_name / config.output_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    prefix = trajectory_dir.name
    qa_jsonl = output_dir / f"{prefix}_qa_pairs.jsonl"
    positions_jsonl = output_dir / f"{prefix}_qa_pairs_positions.jsonl"
    positions_summary = output_dir / f"{prefix}_qa_pairs_positions_summary.jsonl"
    token_report = reports_dir / f"{prefix}_token_report.jsonl"
    decile_report = reports_dir / f"{prefix}_decile_token_report.json"
    bucket_report = reports_dir / f"{prefix}_bucket_token_report.json"

    if not qa_jsonl.exists():
        fallback = ensure_single(output_dir.glob("*_qa_pairs.jsonl"), "qa_pairs file", output_dir)
        if fallback is not None:
            qa_jsonl = fallback
            prefix = qa_jsonl.name[: -len("_qa_pairs.jsonl")]
            positions_jsonl = output_dir / f"{prefix}_qa_pairs_positions.jsonl"
            positions_summary = output_dir / f"{prefix}_qa_pairs_positions_summary.jsonl"
            token_report = reports_dir / f"{prefix}_token_report.jsonl"
            decile_report = reports_dir / f"{prefix}_decile_token_report.json"
            bucket_report = reports_dir / f"{prefix}_bucket_token_report.json"

    qa_missing = force or not qa_jsonl.exists()
    positions_missing = force or not positions_jsonl.exists() or not positions_summary.exists()
    token_missing = (not skip_tokens) and (force or not token_report.exists())
    summary_missing = (not skip_tokens) and (
        force or not decile_report.exists() or not bucket_report.exists()
    )

    if not any([qa_missing, positions_missing, token_missing, summary_missing]):
        print(f"[skip] {batch_name}/{mode}: all expected artifacts already exist")
        return

    needs_token_api((qa_missing and not skip_tokens) or token_missing)

    print(f"[prepare] {batch_name}/{mode}")
    if qa_missing:
        cmd = [
            sys.executable,
            "qa_scripts/qa_cli.py",
            "run-all",
            "--trajectory-dir",
            str(trajectory_dir),
            "--output-dir",
            str(output_dir),
            "--mode",
            config.qa_mode,
            "--model",
            model,
        ]
        if skip_tokens:
            cmd.append("--skip-tokens")
        run_command(cmd, dry_run)
        return

    if positions_missing:
        run_command(
            [
                sys.executable,
                "qa_scripts/qa_cli.py",
                "positions",
                "--input",
                str(qa_jsonl),
                "--output",
                str(positions_jsonl),
                "--summary-output",
                str(positions_summary),
                "--trajectory-root",
                str(trajectory_dir),
            ],
            dry_run,
        )

    if skip_tokens:
        return

    if token_missing:
        run_command(
            [
                sys.executable,
                "qa_scripts/qa_cli.py",
                "token-count",
                "--input",
                str(qa_jsonl),
                "--output",
                str(token_report),
                "--model",
                model,
                "--include-pairs",
            ],
            dry_run,
        )

    if summary_missing or token_missing or positions_missing:
        run_command(
            [
                sys.executable,
                "qa_scripts/qa_cli.py",
                "token-summary",
                "--positions",
                str(positions_jsonl),
                "--token-report",
                str(token_report),
                "--decile-output",
                str(decile_report),
                "--bucket-output",
                str(bucket_report),
            ],
            dry_run,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare qa_pairs artifacts required by rebuild_paper_results_from_raw.py."
    )
    parser.add_argument("--batches", default="1,2,3,4,5")
    parser.add_argument(
        "--modes",
        default="clarify_interact,clarify_interact_v2,interact_with_qmark",
    )
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--force", action="store_true", help="Recompute all QA artifacts.")
    parser.add_argument(
        "--skip-tokens",
        action="store_true",
        help="Skip token-count and token-summary steps.",
    )
    parser.add_argument(
        "--rebuild-after",
        action="store_true",
        help="Run scripts/rebuild_paper_results_from_raw.py after preparation completes.",
    )
    parser.add_argument(
        "--rebuild-out-dir",
        default="paper_reproduction_outputs/from_raw",
        help="Output directory passed through when using --rebuild-after.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    batches = parse_csv_ints(args.batches)
    modes = parse_csv_modes(args.modes)

    for batch in batches:
        for mode in modes:
            prepare_mode_batch(
                batch,
                mode,
                MODE_CONFIGS[mode],
                model=args.model,
                force=args.force,
                skip_tokens=args.skip_tokens,
                dry_run=args.dry_run,
            )

    if args.rebuild_after:
        cmd = [
            sys.executable,
            "scripts/rebuild_paper_results_from_raw.py",
            "--batches",
            args.batches,
            "--out-dir",
            args.rebuild_out_dir,
        ]
        run_command(cmd, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
