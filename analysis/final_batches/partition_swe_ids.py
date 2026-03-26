#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: datasets. Install with `pip install datasets` or use the project's env."
    ) from exc


FIXED_BATCH = [
    "astropy__astropy-12907",
    "astropy__astropy-14369",
    "astropy__astropy-7606",
    "astropy__astropy-7671",
    "django__django-10097",
    "django__django-10554",
    "django__django-10880",
    "django__django-10914",
    "django__django-11095",
    "django__django-11099",
    "django__django-11133",
    "django__django-11206",
    "django__django-11265",
    "django__django-11333",
    "django__django-11603",
    "django__django-11728",
    "django__django-11734",
    "django__django-11848",
    "django__django-12039",
    "django__django-12193",
    "django__django-12209",
    "django__django-12273",
    "django__django-12276",
    "django__django-12304",
    "django__django-12308",
    "django__django-12663",
    "django__django-12713",
    "django__django-13012",
    "django__django-13023",
    "django__django-13212",
    "django__django-13343",
    "django__django-13344",
    "django__django-13417",
    "django__django-13512",
    "django__django-13786",
    "django__django-13933",
    "django__django-14034",
    "django__django-14122",
    "django__django-14140",
    "django__django-14155",
    "django__django-14373",
    "django__django-14500",
    "django__django-14534",
    "django__django-15098",
    "django__django-15277",
    "django__django-15278",
    "django__django-15375",
    "django__django-15569",
    "django__django-15731",
    "django__django-16256",
    "django__django-16315",
    "django__django-16429",
    "django__django-16493",
    "django__django-16527",
    "matplotlib__matplotlib-23476",
    "matplotlib__matplotlib-24627",
    "matplotlib__matplotlib-24637",
    "matplotlib__matplotlib-25332",
    "psf__requests-1142",
    "psf__requests-2931",
    "pydata__xarray-3993",
    "pydata__xarray-6992",
    "pylint-dev__pylint-4551",
    "pylint-dev__pylint-4604",
    "pylint-dev__pylint-4661",
    "pylint-dev__pylint-6386",
    "pytest-dev__pytest-5262",
    "pytest-dev__pytest-5787",
    "pytest-dev__pytest-7324",
    "scikit-learn__scikit-learn-12973",
    "scikit-learn__scikit-learn-13328",
    "scikit-learn__scikit-learn-14087",
    "scikit-learn__scikit-learn-14496",
    "scikit-learn__scikit-learn-14894",
    "scikit-learn__scikit-learn-26194",
    "sphinx-doc__sphinx-10466",
    "sphinx-doc__sphinx-11445",
    "sphinx-doc__sphinx-7462",
    "sphinx-doc__sphinx-8475",
    "sphinx-doc__sphinx-8721",
    "sphinx-doc__sphinx-9281",
    "sphinx-doc__sphinx-9367",
    "sympy__sympy-12489",
    "sympy__sympy-13091",
    "sympy__sympy-13757",
    "sympy__sympy-13798",
    "sympy__sympy-13877",
    "sympy__sympy-15599",
    "sympy__sympy-16597",
    "sympy__sympy-17630",
    "sympy__sympy-18211",
    "sympy__sympy-19495",
    "sympy__sympy-19637",
    "sympy__sympy-19783",
    "sympy__sympy-20916",
    "sympy__sympy-21847",
    "sympy__sympy-22080",
    "sympy__sympy-23950",
    "sympy__sympy-24066",
    "sympy__sympy-24213",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Partition interactive-swe instance_ids into 5 batches of 100."
    )
    parser.add_argument(
        "--dataset",
        default="cmu-lti/interactive-swe",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling remaining ids.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON file path.",
    )
    parser.add_argument(
        "--toml-dir",
        default="",
        help="Optional output directory for batch TOML files (batch1.toml..batch5.toml).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    df = dataset.to_pandas()
    all_ids = df["instance_id"].tolist()

    fixed_set = set(FIXED_BATCH)
    missing = [i for i in FIXED_BATCH if i not in all_ids]
    if missing:
        raise SystemExit(f"Fixed batch contains ids not in dataset: {missing}")

    remaining = [i for i in all_ids if i not in fixed_set]
    if len(FIXED_BATCH) != 100:
        raise SystemExit(f"Fixed batch must have 100 ids, got {len(FIXED_BATCH)}")

    rng = random.Random(args.seed)
    rng.shuffle(remaining)

    needed = 400
    if len(remaining) < needed:
        raise SystemExit(
            f"Not enough remaining ids to form 4 batches: have {len(remaining)}"
        )

    batches = [FIXED_BATCH]
    for i in range(4):
        start = i * 100
        batches.append(remaining[start : start + 100])

    payload = {"selected_ids": batches}
    print(json.dumps(payload, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    if args.toml_dir:
        os.makedirs(args.toml_dir, exist_ok=True)
        for idx, batch in enumerate(batches, start=1):
            toml_path = os.path.join(args.toml_dir, f"batch{idx}.toml")
            with open(toml_path, "w", encoding="utf-8") as f:
                f.write("selected_ids = [\n")
                for item in batch:
                    f.write(f'  "{item}",\n')
                f.write("]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
