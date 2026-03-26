#!/usr/bin/env python3
"""
Unified QA analysis tools for trajectory-to-report workflows.

This script consolidates the legacy scripts under `qa_scripts/` into one CLI:
- extract: build Q/A pairs JSONL (+ optional Markdown) from trajectory JSON files
- positions: annotate Q/A pairs with trajectory position metrics
- token-count: count question/answer tokens via Anthropic's count_tokens API
- token-summary: aggregate token counts by decile and early/mid/late buckets
- asked-overlap: compare asked/not-asked resolved overlaps across reports
- ask-effect: summarize asked-vs-not-asked resolve rates across reports
- prettify: write indented JSON arrays from JSONL files
- run-all: run the standard pipeline and write all major outputs

Examples:
  python qa_scripts/qa_cli.py extract --input-dir final_trajectories_best100/interact --output qa_pairs/interact/interact_qa_pairs.jsonl --markdown
  python qa_scripts/qa_cli.py positions --input qa_pairs/interact/interact_qa_pairs.jsonl --output qa_pairs/interact/interact_qa_pairs_positions.jsonl --summary-output qa_pairs/interact/interact_qa_pairs_positions_summary.jsonl
  python qa_scripts/qa_cli.py prettify --input qa_pairs/interact/interact_qa_pairs.jsonl --output qa_pairs/interact/interact_qa_pairs_pretty.json
  python qa_scripts/qa_cli.py run-all --trajectory-dir final_trajectories_best100/clarify_interact_new_final --output-dir qa_pairs/clarify_interact --mode clarify --reports full=in_progress_outputs/full/report.json hidden=in_progress_outputs/hidden/report.json interact=in_progress_outputs/interact/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Set, Tuple


CATEGORIES = {
    "problem_spec": [
        r"expected",
        r"actual",
        r"error",
        r"stack trace",
        r"stacktrace",
        r"output",
        r"repro",
        r"reproduce",
        r"example",
        r"case",
        r"pattern",
        r"behavior",
        r"bug",
        r"issue",
        r"fails",
        r"failure",
        r"traceback",
    ],
    "tests": [r"test", r"test case", r"pytest", r"unit", r"integration"],
    "environment": [
        r"python",
        r"version",
        r"os",
        r"linux",
        r"macos",
        r"windows",
        r"virtualenv",
        r"conda",
        r"pip",
        r"install",
        r"package",
        r"xdist",
        r"cache",
        r"compiled",
        r"path",
        r"sys\.path",
        r"environment",
    ],
    "implementation": [
        r"function",
        r"method",
        r"class",
        r"file",
        r"line",
        r"code",
        r"logic",
        r"behavior",
        r"change",
        r"fix",
        r"handle",
        r"update",
        r"implement",
        r"apply",
    ],
    "scope": [r"should i", r"do i need", r"only", r"or", r"both", r"scope", r"focus", r"confirm"],
    "confirmation": [r"okay", r"correct", r"confirm", r"right", r"expected\?", r"is it"],
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def is_clarify_question(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("source") != "agent" or item.get("action") != "message":
        return False
    meta = item.get("tool_call_metadata")
    return isinstance(meta, dict) and meta.get("function_name") == "clarify"


def is_interact_question(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("source") != "agent" or item.get("action") != "message":
        return False
    args = item.get("args") or {}
    return isinstance(args, dict) and args.get("wait_for_response") is True


def is_user_message(item: dict) -> bool:
    return (
        isinstance(item, dict)
        and item.get("source") == "user"
        and item.get("action") == "message"
    )


def extract_pairs(items: list[dict], mode: str, question_contains: Optional[str]) -> list[dict]:
    pairs: list[dict] = []
    i = 0
    while i < len(items):
        item = items[i]
        is_question = is_clarify_question(item) if mode == "clarify" else is_interact_question(item)
        if is_question:
            message = item.get("message") or ""
            if question_contains and question_contains not in message:
                i += 1
                continue
            question = {
                "id": item.get("id"),
                "timestamp": item.get("timestamp"),
                "message": message,
            }
            answer = None
            for j in range(i + 1, len(items)):
                if is_user_message(items[j]):
                    answer = {
                        "id": items[j].get("id"),
                        "timestamp": items[j].get("timestamp"),
                        "message": items[j].get("message"),
                    }
                    i = j
                    break
            pairs.append({"question": question, "answer": answer})
        i += 1
    return pairs


def fence_for_text(text: str) -> str:
    if text is None:
        return "```"
    max_len = 0
    for match in re.finditer(r"`+", text):
        max_len = max(max_len, len(match.group(0)))
    return "`" * max(3, max_len + 1)


def write_markdown(path: Path, records: list[dict]) -> None:
    lines: list[str] = []
    for record in records:
        trajectory = record["trajectory"]
        pairs = record["pairs"]
        lines.append(f"## {trajectory}")
        for idx, pair in enumerate(pairs, 1):
            q = pair["question"]
            a = pair["answer"] or {}
            lines.append(f"{idx}) Q ({q.get('timestamp')}):")
            q_fence = fence_for_text(q.get("message"))
            lines.append(f"{q_fence}text")
            lines.append(q.get("message") or "")
            lines.append(q_fence)
            lines.append(f"A ({a.get('timestamp')}):")
            a_fence = fence_for_text(a.get("message"))
            lines.append(f"{a_fence}text")
            lines.append(a.get("message") or "")
            lines.append(a_fence)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def cmd_extract(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    markdown_path = Path(args.markdown_output) if args.markdown_output else None

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    records: list[dict] = []
    for path in sorted(input_dir.rglob("*.json")):
        try:
            data = load_json(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        pairs = extract_pairs(data, args.mode, args.question_contains)
        if not pairs:
            continue
        records.append(
            {
                "trajectory": path.name,
                "trajectory_path": str(path),
                "pairs": pairs,
            }
        )

    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} trajectories to {output_path}")
    if args.markdown:
        md_target = markdown_path or output_path.with_suffix(".md")
        write_markdown(md_target, records)
        print(f"Wrote Markdown report to {md_target}")
    return 0


def load_trajectory(path: Path) -> Tuple[list[dict], Dict[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Trajectory is not a list: {path}")
    id_to_index: Dict[int, int] = {}
    for idx, item in enumerate(data):
        if isinstance(item, dict) and isinstance(item.get("id"), int):
            id_to_index[item["id"]] = idx
    return data, id_to_index


def is_intent_bookkeeping(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    # Only remove clarify-intent bookkeeping emitted by the main agent.
    # Environment, user, and ordinary agent/tool events remain in the stream.
    if item.get("source") != "agent":
        return False
    if item.get("action") in {"delegate", "intent_decision"}:
        return True
    if item.get("observation") == "delegate":
        return True
    return False


def build_filtered_id_to_index(data: list[dict]) -> Dict[int, int]:
    id_to_index: Dict[int, int] = {}
    filtered_idx = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        if is_intent_bookkeeping(item):
            continue
        id_to_index[item_id] = filtered_idx
        filtered_idx += 1
    return id_to_index


def decile_for_fraction(frac: Optional[float]) -> Optional[int]:
    if frac is None:
        return None
    if frac >= 1.0:
        return 10
    if frac < 0:
        return 1
    return int(frac * 10) + 1


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "avg": 0, "min": 0, "max": 0}
    return {
        "count": len(values),
        "avg": round(mean(values), 2),
        "min": min(values),
        "max": max(values),
    }


def resolve_trajectory_path(record: dict, trajectory_root: Optional[Path]) -> Optional[Path]:
    path_value = record.get("trajectory_path")
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path
    trajectory_name = record.get("trajectory")
    if trajectory_root and trajectory_name:
        candidate = trajectory_root / trajectory_name
        if candidate.exists():
            return candidate
    return None


def process_positions_file(
    qa_path: Path,
    trajectory_root: Optional[Path],
    include_messages: bool,
) -> Tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summaries: list[dict] = []
    trajectory_cache: Dict[Path, Tuple[list[dict], Dict[int, int]]] = {}

    for record in load_jsonl(qa_path):
        pairs = record.get("pairs") or []
        trajectory_path = resolve_trajectory_path(record, trajectory_root)
        id_to_index: Dict[int, int] = {}
        id_to_index_no_intent: Dict[int, int] = {}
        event_count = 0
        filtered_event_count = 0
        min_id = None
        max_id = None

        if trajectory_path:
            if trajectory_path not in trajectory_cache:
                trajectory_cache[trajectory_path] = load_trajectory(trajectory_path)
            trajectory_data, id_to_index = trajectory_cache[trajectory_path]
            event_count = len(trajectory_data)
            id_to_index_no_intent = build_filtered_id_to_index(trajectory_data)
            filtered_event_count = len(id_to_index_no_intent)
            if event_count:
                ids = [item.get("id") for item in trajectory_data if isinstance(item, dict)]
                ids = [value for value in ids if isinstance(value, int)]
                if ids:
                    min_id = min(ids)
                    max_id = max(ids)

        for pair in pairs:
            question = pair.get("question") or {}
            answer = pair.get("answer") or {}
            question_id = question.get("id")
            question_index = None
            question_index_no_intent = None
            if isinstance(question_id, int) and question_id in id_to_index:
                question_index = id_to_index[question_id]
            if isinstance(question_id, int) and question_id in id_to_index_no_intent:
                question_index_no_intent = id_to_index_no_intent[question_id]

            id_frac = None
            if isinstance(question_id, int) and isinstance(max_id, int) and max_id > 0:
                id_frac = question_id / max_id

            index_frac = None
            if isinstance(question_index, int) and event_count > 0:
                index_frac = (question_index + 1) / event_count

            index_no_intent_frac = None
            if isinstance(question_index_no_intent, int) and filtered_event_count > 0:
                index_no_intent_frac = (question_index_no_intent + 1) / filtered_event_count

            row = {
                "source_file": qa_path.name,
                "trajectory": record.get("trajectory"),
                "trajectory_path": str(trajectory_path) if trajectory_path else None,
                "trajectory_event_count": event_count,
                "trajectory_event_count_no_intent": filtered_event_count,
                "trajectory_min_id": min_id,
                "trajectory_max_id": max_id,
                "question_id": question_id,
                "question_index": question_index,
                "question_index_no_intent": question_index_no_intent,
                "question_position_id_frac": id_frac,
                "question_position_index_frac": index_frac,
                "question_position_index_no_intent_frac": index_no_intent_frac,
                "question_decile_id": decile_for_fraction(id_frac),
                "question_decile_index": decile_for_fraction(index_frac),
                "question_decile_index_no_intent": decile_for_fraction(index_no_intent_frac),
                "question_timestamp": question.get("timestamp"),
                "answer_id": answer.get("id"),
                "answer_timestamp": answer.get("timestamp"),
            }
            if include_messages:
                row["question_message"] = question.get("message")
                row["answer_message"] = answer.get("message")
            rows.append(row)

        if pairs:
            summaries.append(
                {
                    "source_file": qa_path.name,
                    "trajectory": record.get("trajectory"),
                    "trajectory_path": str(trajectory_path) if trajectory_path else None,
                    "trajectory_event_count": event_count,
                    "trajectory_event_count_no_intent": filtered_event_count,
                    "trajectory_min_id": min_id,
                    "trajectory_max_id": max_id,
                    "questions": len(pairs),
                }
            )

    return rows, summaries


def cmd_positions(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else None
    trajectory_root = Path(args.trajectory_root) if args.trajectory_root else None
    include_messages = not args.no_messages

    qa_files = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    all_rows: list[dict] = []
    all_summaries: list[dict] = []
    for qa_file in qa_files:
        rows, summaries = process_positions_file(qa_file, trajectory_root, include_messages)
        all_rows.extend(rows)
        all_summaries.extend(summaries)

    write_jsonl(output_path, all_rows)
    print(f"Wrote {len(all_rows)} position rows to {output_path}")
    if summary_path:
        write_jsonl(summary_path, all_summaries)
        print(f"Wrote {len(all_summaries)} trajectory summaries to {summary_path}")
    if all_rows and include_messages:
        lengths = [len(row.get("question_message") or "") for row in all_rows]
        summary = summarize_lengths(lengths)
        print(f"question_message_lengths={summary}")
    return 0


def get_anthropic_client():
    from anthropic import Anthropic  # type: ignore

    return Anthropic()


def count_tokens(client, model: str, text: str, role: str) -> int:
    if not text:
        return 0
    messages = [{"role": role, "content": text}]
    if hasattr(client, "messages") and hasattr(client.messages, "count_tokens"):
        resp = client.messages.count_tokens(model=model, messages=messages)
        if hasattr(resp, "input_tokens"):
            return resp.input_tokens
        if isinstance(resp, dict):
            return resp.get("input_tokens", 0)
        raise RuntimeError("count_tokens() response missing input_tokens.")
    if hasattr(client, "beta") and hasattr(client.beta, "messages") and hasattr(client.beta.messages, "count_tokens"):
        resp = client.beta.messages.count_tokens(model=model, messages=messages)
        if hasattr(resp, "input_tokens"):
            return resp.input_tokens
        if isinstance(resp, dict):
            return resp.get("input_tokens", 0)
        raise RuntimeError("count_tokens() response missing input_tokens.")
    raise RuntimeError("Anthropic client does not support count_tokens().")


def summarize_counts(values: list[int]) -> dict:
    if not values:
        return {"total": 0, "avg": 0, "min": 0, "max": 0}
    return {
        "total": sum(values),
        "avg": round(mean(values), 2),
        "min": min(values),
        "max": max(values),
    }


def process_token_file(client, model: str, path: Path, include_pairs: bool) -> list[dict]:
    records: list[dict] = []
    file_question_counts: list[int] = []
    file_answer_counts: list[int] = []
    file_missing_answers = 0

    for record in load_jsonl(path):
        pairs = record.get("pairs") or []
        question_counts: list[int] = []
        answer_counts: list[int] = []
        pair_token_counts: list[dict] = []

        for pair in pairs:
            question_text = ((pair.get("question") or {}).get("message") or "").rstrip()
            answer_text = ((pair.get("answer") or {}).get("message") or "").rstrip()
            q_tokens = count_tokens(client, model, question_text, "user")
            a_tokens = count_tokens(client, model, answer_text, "assistant")
            question_counts.append(q_tokens)
            answer_counts.append(a_tokens)
            if include_pairs:
                pair_token_counts.append(
                    {
                        "question_id": (pair.get("question") or {}).get("id"),
                        "question_timestamp": (pair.get("question") or {}).get("timestamp"),
                        "answer_id": (pair.get("answer") or {}).get("id"),
                        "answer_timestamp": (pair.get("answer") or {}).get("timestamp"),
                        "question_tokens": q_tokens,
                        "answer_tokens": a_tokens,
                    }
                )
            if not answer_text:
                file_missing_answers += 1

        file_question_counts.extend(question_counts)
        file_answer_counts.extend(answer_counts)

        q_summary = summarize_counts(question_counts)
        a_summary = summarize_counts(answer_counts)
        trajectory_record = {
            "type": "trajectory",
            "source_file": path.name,
            "trajectory": record.get("trajectory"),
            "trajectory_path": record.get("trajectory_path"),
            "num_questions": len(pairs),
            "missing_answers": sum(1 for pair in pairs if not (pair.get("answer") or {}).get("message")),
            "question_tokens_total": q_summary["total"],
            "question_tokens_avg": q_summary["avg"],
            "question_tokens_min": q_summary["min"],
            "question_tokens_max": q_summary["max"],
            "answer_tokens_total": a_summary["total"],
            "answer_tokens_avg": a_summary["avg"],
            "answer_tokens_min": a_summary["min"],
            "answer_tokens_max": a_summary["max"],
        }
        if include_pairs:
            trajectory_record["pair_token_counts"] = pair_token_counts
        records.append(trajectory_record)

    avg_questions = round(len(file_question_counts) / len(records), 2) if records else 0
    file_summary = {
        "type": "file_summary",
        "source_file": path.name,
        "trajectories": len(records),
        "questions_total": len(file_question_counts),
        "num_questions": len(file_question_counts),
        "avg_questions_per_trajectory": avg_questions,
        "missing_answers": file_missing_answers,
    }
    file_summary.update(
        {
            "question_tokens_total": summarize_counts(file_question_counts)["total"],
            "question_tokens_avg": summarize_counts(file_question_counts)["avg"],
            "question_tokens_min": summarize_counts(file_question_counts)["min"],
            "question_tokens_max": summarize_counts(file_question_counts)["max"],
            "answer_tokens_total": summarize_counts(file_answer_counts)["total"],
            "answer_tokens_avg": summarize_counts(file_answer_counts)["avg"],
            "answer_tokens_min": summarize_counts(file_answer_counts)["min"],
            "answer_tokens_max": summarize_counts(file_answer_counts)["max"],
        }
    )
    return [file_summary] + records


def cmd_token_count(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    jsonl_files = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]

    client = get_anthropic_client()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for jsonl_path in jsonl_files:
            records = process_token_file(client, args.model, jsonl_path, args.include_pairs)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Wrote token report to {output_path}")
    return 0


def build_decile_index(positions_path: Path, source: str) -> Dict[Tuple[str, int], int]:
    index: Dict[Tuple[str, int], int] = {}
    for row in load_jsonl(positions_path):
        key = (row.get("trajectory"), row.get("question_id"))
        if source == "id":
            decile = row.get("question_decile_id")
        elif source == "index_no_intent":
            decile = row.get("question_decile_index_no_intent")
        else:
            decile = row.get("question_decile_index")
        if key[0] and isinstance(key[1], int) and isinstance(decile, int):
            index[key] = decile
    return index


def build_message_index(positions_path: Path) -> Dict[Tuple[str, int], str]:
    index: Dict[Tuple[str, int], str] = {}
    for row in load_jsonl(positions_path):
        key = (row.get("trajectory"), row.get("question_id"))
        if key[0] and isinstance(key[1], int):
            index[key] = row.get("question_message") or ""
    return index


def compile_categories() -> Dict[str, list[re.Pattern]]:
    return {k: [re.compile(p, re.IGNORECASE) for p in pats] for k, pats in CATEGORIES.items()}


def categorize(text: str, patterns: Dict[str, list[re.Pattern]]) -> list[str]:
    matches = []
    for cat, pats in patterns.items():
        if any(p.search(text) for p in pats):
            matches.append(cat)
    return matches


def summarize_deciles(token_report: Path, deciles: Dict[Tuple[str, int], int]) -> list[dict]:
    agg = defaultdict(lambda: {"questions": 0, "q_tokens": 0, "a_tokens": 0})
    for record in load_jsonl(token_report):
        if record.get("type") != "trajectory":
            continue
        trajectory = record.get("trajectory")
        for pair in record.get("pair_token_counts", []):
            question_id = pair.get("question_id")
            decile = deciles.get((trajectory, question_id))
            if not isinstance(decile, int):
                continue
            agg[decile]["questions"] += 1
            agg[decile]["q_tokens"] += pair.get("question_tokens", 0)
            agg[decile]["a_tokens"] += pair.get("answer_tokens", 0)
    summary = []
    for decile in sorted(agg):
        count = agg[decile]["questions"]
        q_tokens = agg[decile]["q_tokens"]
        a_tokens = agg[decile]["a_tokens"]
        summary.append(
            {
                "decile": decile,
                "questions": count,
                "q_tokens_total": q_tokens,
                "q_tokens_avg": round(q_tokens / count, 2) if count else 0,
                "a_tokens_total": a_tokens,
                "a_tokens_avg": round(a_tokens / count, 2) if count else 0,
            }
        )
    return summary


def summarize_buckets(
    token_report: Path,
    deciles: Dict[Tuple[str, int], int],
    messages: Dict[Tuple[str, int], str],
    patterns: Dict[str, list[re.Pattern]],
    include_categories: bool,
) -> list[dict]:
    buckets = {"early": set(range(1, 4)), "mid": set(range(4, 8)), "late": set(range(8, 11))}
    agg = defaultdict(lambda: defaultdict(lambda: {"questions": 0, "q_tokens": 0, "a_tokens": 0}))
    for record in load_jsonl(token_report):
        if record.get("type") != "trajectory":
            continue
        trajectory = record.get("trajectory")
        for pair in record.get("pair_token_counts", []):
            question_id = pair.get("question_id")
            key = (trajectory, question_id)
            decile = deciles.get(key)
            if not isinstance(decile, int):
                continue
            bucket = next((name for name, decs in buckets.items() if decile in decs), None)
            if not bucket:
                continue
            categories = ["all"]
            if include_categories:
                categories = categorize(messages.get(key, ""), patterns) or ["other"]
            for cat in categories:
                agg[bucket][cat]["questions"] += 1
                agg[bucket][cat]["q_tokens"] += pair.get("question_tokens", 0)
                agg[bucket][cat]["a_tokens"] += pair.get("answer_tokens", 0)
    summary = []
    for bucket in ["early", "mid", "late"]:
        for cat, values in sorted(agg.get(bucket, {}).items()):
            count = values["questions"]
            q_tokens = values["q_tokens"]
            a_tokens = values["a_tokens"]
            summary.append(
                {
                    "bucket": bucket,
                    "category": cat,
                    "questions": count,
                    "q_tokens_total": q_tokens,
                    "q_tokens_avg": round(q_tokens / count, 2) if count else 0,
                    "a_tokens_total": a_tokens,
                    "a_tokens_avg": round(a_tokens / count, 2) if count else 0,
                }
            )
    return summary


def cmd_token_summary(args: argparse.Namespace) -> int:
    positions_path = Path(args.positions)
    token_report = Path(args.token_report)
    decile_out = Path(args.decile_output)
    bucket_out = Path(args.bucket_output)

    deciles = build_decile_index(positions_path, args.decile_source)
    messages = build_message_index(positions_path)
    patterns = compile_categories()
    decile_summary = summarize_deciles(token_report, deciles)
    bucket_summary = summarize_buckets(
        token_report,
        deciles,
        messages,
        patterns,
        args.include_categories,
    )
    write_json(decile_out, decile_summary)
    write_json(bucket_out, bucket_summary)
    print(f"Wrote decile summary to {decile_out}")
    print(f"Wrote bucket summary to {bucket_out}")
    return 0


def extract_instance_id(trajectory_name: str) -> str:
    name = trajectory_name[:-5] if trajectory_name.endswith(".json") else trajectory_name
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name


def asked_ids_from_qa_pairs(path: Path) -> Set[str]:
    asked: Set[str] = set()
    for record in load_jsonl(path):
        trajectory = record.get("trajectory") or ""
        if trajectory:
            asked.add(extract_instance_id(trajectory))
    return asked


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_population(report: dict, population: str) -> Set[str]:
    if population == "submitted":
        return set(report.get("submitted_ids") or [])
    completed = set(report.get("completed_ids") or [])
    if completed:
        return completed
    return set(report.get("submitted_ids") or [])


def parse_reports(values: List[str]) -> List[Tuple[str, Path]]:
    reports = []
    for value in values:
        if "=" in value:
            name, raw = value.split("=", 1)
        else:
            raw = value
            path = Path(raw)
            name = path.parent.name or path.stem
        reports.append((name, Path(raw)))
    return reports


def cmd_asked_overlap(args: argparse.Namespace) -> int:
    qa_pairs = Path(args.qa_pairs)
    output = Path(args.output)
    asked_ids = asked_ids_from_qa_pairs(qa_pairs)
    reports = parse_reports(args.reports)
    if not reports:
        print("No reports provided.", file=sys.stderr)
        return 1

    primary_name, primary_path = reports[0]
    primary = load_report(primary_path)
    population = get_population(primary, args.population)
    resolved_primary = set(primary.get("resolved_ids") or [])

    asked = population & asked_ids
    not_asked = population - asked_ids
    subsets: Dict[str, list[str]] = {}
    if args.mode in ("asked", "both"):
        subsets["asked"] = sorted(asked & resolved_primary)
    if args.mode in ("not_asked", "both"):
        subsets["not_asked"] = sorted(not_asked & resolved_primary)

    payload: Dict[str, dict] = {
        "primary_report": primary_name,
        "population": args.population,
        "asked_ids_total": len(asked_ids),
        "subsets": {name: values for name, values in subsets.items()},
        "overlaps": {},
    }
    for name, path in reports[1:]:
        other = load_report(path)
        resolved_other = set(other.get("resolved_ids") or [])
        overlap = {}
        for subset_name, values in subsets.items():
            subset_set = set(values)
            overlap[subset_name] = {
                "resolved": sorted(subset_set & resolved_other),
                "unresolved": sorted(subset_set - resolved_other),
            }
        payload["overlaps"][name] = overlap

    write_json(output, payload)
    print(f"Wrote overlap summary to {output}")
    return 0


def resolve_rate(resolved: Set[str], population: Set[str]) -> float:
    if not population:
        return 0.0
    return round(len(resolved & population) / len(population) * 100, 2)


def summarize_asked_effect(asked_ids: Set[str], report: dict, population: str) -> dict:
    pop = get_population(report, population)
    resolved = set(report.get("resolved_ids") or [])
    asked = asked_ids & pop
    not_asked = pop - asked_ids
    return {
        "population_total": len(pop),
        "asked_total": len(asked),
        "asked_resolved": len(asked & resolved),
        "asked_rate": resolve_rate(resolved, asked),
        "not_asked_total": len(not_asked),
        "not_asked_resolved": len(not_asked & resolved),
        "not_asked_rate": resolve_rate(resolved, not_asked),
    }


def cmd_ask_effect(args: argparse.Namespace) -> int:
    qa_pairs = Path(args.qa_pairs)
    asked_ids = asked_ids_from_qa_pairs(qa_pairs)
    reports = parse_reports(args.reports)
    output = Path(args.output)

    results: Dict[str, dict] = {}
    for name, path in reports:
        report = load_report(path)
        results[name] = summarize_asked_effect(asked_ids, report, args.population)

    write_json(output, results)
    if not args.no_print:
        print(f"Wrote summary to {output}")
        for name, stats in results.items():
            print(f"\n{name}:")
            print(f"  population_total: {stats['population_total']}")
            print("  asked: {asked_total} resolved {asked_resolved} rate {asked_rate}".format(**stats))
            print(
                "  not_asked: {not_asked_total} resolved {not_asked_resolved} rate {not_asked_rate}".format(
                    **stats
                )
            )
    return 0


def cmd_prettify(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = list(load_jsonl(input_path))
    write_json(output_path, rows)
    print(f"Wrote {len(rows)} records to {output_path}")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or trajectory_dir.name
    suffix = args.suffix or ""
    base = f"{prefix}{suffix}"

    qa_jsonl = output_dir / f"{base}_qa_pairs.jsonl"
    qa_md = output_dir / f"{base}_qa_pairs.md"
    qa_pretty = output_dir / f"{base}_qa_pairs_pretty.json"
    positions_jsonl = output_dir / f"{base}_qa_pairs_positions.jsonl"
    positions_summary = output_dir / f"{base}_qa_pairs_positions_summary.jsonl"
    positions_pretty = output_dir / f"{base}_qa_pairs_positions_pretty.json"
    token_report = reports_dir / f"{base}_token_report.jsonl"
    decile_report = reports_dir / f"{base}_decile_token_report.json"
    bucket_report = reports_dir / f"{base}_bucket_token_report.json"
    overlap_report = output_dir / "asked_not_asked_overlap_summary.json"
    effect_report = output_dir / "asked_vs_not_asked_summary.json"

    status = 0

    print("== extract ==")
    status = max(
        status,
        cmd_extract(
            argparse.Namespace(
                input_dir=str(trajectory_dir),
                output=str(qa_jsonl),
                markdown=args.markdown,
                markdown_output=str(qa_md),
                mode=args.mode,
                question_contains=args.question_contains,
            )
        ),
    )
    print()

    print("== prettify (qa pairs) ==")
    status = max(
        status,
        cmd_prettify(argparse.Namespace(input=str(qa_jsonl), output=str(qa_pretty))),
    )
    print()

    print("== positions ==")
    status = max(
        status,
        cmd_positions(
            argparse.Namespace(
                input=str(qa_jsonl),
                output=str(positions_jsonl),
                summary_output=str(positions_summary),
                trajectory_root=str(args.trajectory_root or trajectory_dir),
                no_messages=args.no_messages,
            )
        ),
    )
    print()

    print("== prettify (positions) ==")
    status = max(
        status,
        cmd_prettify(argparse.Namespace(input=str(positions_jsonl), output=str(positions_pretty))),
    )
    print()

    if not args.skip_tokens:
        print("== token-count ==")
        status = max(
            status,
            cmd_token_count(
                argparse.Namespace(
                    input=str(qa_jsonl),
                    output=str(token_report),
                    model=args.model,
                    include_pairs=True,
                )
            ),
        )
        print()

        print("== token-summary ==")
        status = max(
            status,
            cmd_token_summary(
                argparse.Namespace(
                    positions=str(positions_jsonl),
                    token_report=str(token_report),
                    decile_output=str(decile_report),
                    bucket_output=str(bucket_report),
                    decile_source=args.decile_source,
                    include_categories=args.include_categories,
                )
            ),
        )
        print()

    parsed_reports = parse_reports(args.reports) if args.reports else []
    if parsed_reports:
        report_values = [f"{name}={path}" for name, path in parsed_reports]
        if not args.skip_overlap:
            print("== asked-overlap ==")
            status = max(
                status,
                cmd_asked_overlap(
                    argparse.Namespace(
                        qa_pairs=str(qa_jsonl),
                        reports=report_values,
                        population=args.population,
                        mode="both",
                        output=str(overlap_report),
                    )
                ),
            )
            print()
        if not args.skip_ask_effect:
            print("== ask-effect ==")
            status = max(
                status,
                cmd_ask_effect(
                    argparse.Namespace(
                        qa_pairs=str(qa_jsonl),
                        reports=report_values,
                        population=args.population,
                        output=str(effect_report),
                        no_print=args.no_print,
                    )
                ),
            )
            print()

    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified QA analysis tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_extract = subparsers.add_parser("extract", help="Extract Q/A pairs from trajectory JSON files.")
    p_extract.add_argument("--input-dir", required=True)
    p_extract.add_argument("--output", required=True, help="Output QA-pairs JSONL path.")
    p_extract.add_argument("--mode", choices=["clarify", "interact"], default="clarify")
    p_extract.add_argument("--question-contains")
    p_extract.add_argument("--markdown", action="store_true")
    p_extract.add_argument("--markdown-output", help="Optional Markdown output path.")
    p_extract.set_defaults(func=cmd_extract)

    p_positions = subparsers.add_parser("positions", help="Annotate QA-pairs JSONL with position metrics.")
    p_positions.add_argument("--input", required=True, help="QA-pairs JSONL path or directory of JSONL files.")
    p_positions.add_argument("--output", required=True, help="Output positions JSONL path.")
    p_positions.add_argument("--summary-output", help="Optional output JSONL path for per-trajectory summaries.")
    p_positions.add_argument("--trajectory-root", help="Fallback root to resolve trajectory files by name.")
    p_positions.add_argument("--no-messages", action="store_true", help="Omit question/answer messages.")
    p_positions.set_defaults(func=cmd_positions)

    p_tokens = subparsers.add_parser("token-count", help="Count question/answer tokens for QA-pairs JSONL.")
    p_tokens.add_argument("--input", required=True, help="QA-pairs JSONL path or directory of JSONL files.")
    p_tokens.add_argument("--output", required=True, help="Output token report JSONL path.")
    p_tokens.add_argument("--model", default="claude-sonnet-4-5-20250929")
    p_tokens.add_argument("--include-pairs", action="store_true", help="Include per-pair token counts.")
    p_tokens.set_defaults(func=cmd_token_count, include_pairs=False)

    p_summary = subparsers.add_parser("token-summary", help="Summarize token report by deciles and buckets.")
    p_summary.add_argument("--positions", required=True, help="Positions JSONL input.")
    p_summary.add_argument("--token-report", required=True, help="Token report JSONL from token-count.")
    p_summary.add_argument("--decile-output", required=True, help="Output JSON for decile summary.")
    p_summary.add_argument("--bucket-output", required=True, help="Output JSON for bucket summary.")
    p_summary.add_argument("--decile-source", choices=["index", "id", "index_no_intent"], default="index")
    p_summary.add_argument("--include-categories", action="store_true")
    p_summary.set_defaults(func=cmd_token_summary)

    p_overlap = subparsers.add_parser("asked-overlap", help="Compare asked/not-asked overlap across reports.")
    p_overlap.add_argument("--qa-pairs", required=True, help="QA-pairs JSONL input.")
    p_overlap.add_argument("--reports", nargs="+", required=True, help="Report JSON files (name=path optional).")
    p_overlap.add_argument("--population", choices=["completed", "submitted"], default="completed")
    p_overlap.add_argument("--mode", choices=["asked", "not_asked", "both"], default="both")
    p_overlap.add_argument("--output", default="asked_not_asked_overlap_ids.json")
    p_overlap.set_defaults(func=cmd_asked_overlap)

    p_effect = subparsers.add_parser("ask-effect", help="Summarize asked-vs-not-asked resolve rates.")
    p_effect.add_argument("--qa-pairs", required=True, help="QA-pairs JSONL input.")
    p_effect.add_argument("--reports", nargs="+", required=True, help="Report JSON files (name=path optional).")
    p_effect.add_argument("--population", choices=["completed", "submitted"], default="completed")
    p_effect.add_argument("--output", default="asked_vs_not_asked_summary.json")
    p_effect.add_argument("--no-print", action="store_true")
    p_effect.set_defaults(func=cmd_ask_effect)

    p_pretty = subparsers.add_parser("prettify", help="Convert JSONL records to an indented JSON array.")
    p_pretty.add_argument("--input", required=True, help="Input JSONL file.")
    p_pretty.add_argument("--output", required=True, help="Output pretty JSON file.")
    p_pretty.set_defaults(func=cmd_prettify)

    p_run = subparsers.add_parser("run-all", help="Run the consolidated QA pipeline end-to-end.")
    p_run.add_argument("--trajectory-dir", required=True, help="Directory of trajectory JSON files.")
    p_run.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    p_run.add_argument("--mode", choices=["clarify", "interact"], default="clarify")
    p_run.add_argument("--prefix", help="Optional filename prefix. Defaults to trajectory-dir basename.")
    p_run.add_argument("--suffix", default="", help="Optional suffix added to output basenames.")
    p_run.add_argument("--question-contains")
    p_run.add_argument("--markdown", action="store_true")
    p_run.add_argument("--trajectory-root", help="Optional trajectory root for positions fallback.")
    p_run.add_argument("--no-messages", action="store_true")
    p_run.add_argument("--skip-tokens", action="store_true")
    p_run.add_argument("--model", default="claude-sonnet-4-5-20250929")
    p_run.add_argument("--decile-source", choices=["index", "id", "index_no_intent"], default="index")
    p_run.add_argument("--include-categories", action="store_true")
    p_run.add_argument("--reports", nargs="*", help="Optional report JSON files (name=path optional).")
    p_run.add_argument("--population", choices=["completed", "submitted"], default="completed")
    p_run.add_argument("--skip-overlap", action="store_true")
    p_run.add_argument("--skip-ask-effect", action="store_true")
    p_run.add_argument("--no-print", action="store_true")
    p_run.set_defaults(func=cmd_run_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
