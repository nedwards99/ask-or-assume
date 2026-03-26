#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


DIFFICULTY_TO_ORDINAL = {
    "<15 min fix": 1,
    "15 min - 1 hour": 2,
    "1-4 hours": 3,
    ">4 hours": 4,
}


def read_difficulty_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expand_binary_samples(rows: List[dict]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        difficulty = row["difficulty"].strip()
        if difficulty not in DIFFICULTY_TO_ORDINAL:
            continue
        x = float(DIFFICULTY_TO_ORDINAL[difficulty])
        total = int(row["total"])
        asked = int(row["asked"])
        not_asked = total - asked
        xs.extend([x] * asked)
        ys.extend([1.0] * asked)
        xs.extend([x] * not_asked)
        ys.extend([0.0] * not_asked)
    return xs, ys


def pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def corr_p_value_normal_approx(r: float, n: int) -> Tuple[float, float]:
    if n < 3:
        return 0.0, 1.0
    denom = 1.0 - (r * r)
    if denom <= 0:
        return 0.0, 0.0
    t = r * math.sqrt((n - 2) / denom)
    p = 2.0 * (1.0 - normal_cdf(abs(t)))
    return t, p


def load_asked_totals(path: Path, mode: str, setting: str) -> Tuple[int, int]:
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("mode") == mode and row.get("setting") == setting:
                return int(row["asked_total"]), int(row["population_total"])
    raise RuntimeError(f"No row found in {path} for mode={mode}, setting={setting}")


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float, float]:
    p1 = x1 / n1
    p2 = x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * ((1 / n1) + (1 / n2)))
    if se == 0:
        return p1, p2, 1.0
    z = (p2 - p1) / se
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return p1, p2, p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute ask-vs-difficulty significance and ask-rate difference between two modes."
    )
    parser.add_argument("--difficulty-csv-a", required=True)
    parser.add_argument("--difficulty-csv-b", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--asked-summary-a", required=True)
    parser.add_argument("--asked-summary-b", required=True)
    parser.add_argument("--mode-a", required=True)
    parser.add_argument("--mode-b", required=True)
    parser.add_argument("--setting", default="clarify")
    args = parser.parse_args()

    rows_a = read_difficulty_rows(Path(args.difficulty_csv_a))
    rows_b = read_difficulty_rows(Path(args.difficulty_csv_b))

    xs_a, ys_a = expand_binary_samples(rows_a)
    xs_b, ys_b = expand_binary_samples(rows_b)

    r_a = pearson_r(xs_a, ys_a)
    r_b = pearson_r(xs_b, ys_b)
    t_a, p_a = corr_p_value_normal_approx(r_a, len(xs_a))
    t_b, p_b = corr_p_value_normal_approx(r_b, len(xs_b))

    asked_a, pop_a = load_asked_totals(Path(args.asked_summary_a), args.mode_a, args.setting)
    asked_b, pop_b = load_asked_totals(Path(args.asked_summary_b), args.mode_b, args.setting)
    rate_a, rate_b, p_diff = two_proportion_z_test(asked_a, pop_a, asked_b, pop_b)

    print("Ask-vs-Difficulty Correlation (normal-approx p-value)")
    print(f"- {args.label_a}: n={len(xs_a)}, r={r_a:.4f}, t={t_a:.4f}, p={p_a:.6f}")
    print(f"- {args.label_b}: n={len(xs_b)}, r={r_b:.4f}, t={t_b:.4f}, p={p_b:.6f}")
    print("")
    print("Ask-Rate Difference (two-proportion z-test, two-sided)")
    print(
        f"- {args.label_a}: asked={asked_a}/{pop_a} ({rate_a*100:.2f}%)"
    )
    print(
        f"- {args.label_b}: asked={asked_b}/{pop_b} ({rate_b*100:.2f}%)"
    )
    print(f"- Delta ({args.label_b} - {args.label_a}) = {(rate_b-rate_a)*100:.2f} pp, p={p_diff:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
