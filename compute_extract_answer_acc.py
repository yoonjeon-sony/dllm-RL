"""
Compute _extract_answer accuracy for all sample-results pairs under
outputs/eval_generate_logs.

For each *_results.json, finds the matching *_samples_*.jsonl in the same
directory, extracts <answer>...</answer> from each resp, compares to target,
and writes:
  results[task]["{primary_metric},_extract_answer"]
  results[task]["{primary_metric}_stderr,_extract_answer"]
back into the results file.
"""

import json
import math
import re
import sys
from pathlib import Path


def extract_answer(resp: str) -> str | None:
    """Return content inside the first <answer>...</answer> tag, or None."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", resp, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def normalize(s: str) -> str:
    """Lowercase, strip whitespace, strip surrounding parentheses."""
    return s.strip().lower().strip("()")


def compute_accuracy(samples_path: Path) -> tuple[float, float]:
    """Return (accuracy, stderr) based on <answer> tag extraction."""
    with open(samples_path) as f:
        samples = [json.loads(line) for line in f]

    total = len(samples)
    correct = 0

    for s in samples:
        resp = s["resps"][0][0] if s["resps"] and s["resps"][0] else ""
        target = str(s["target"])

        extracted = extract_answer(resp)
        if extracted is not None and normalize(extracted) == normalize(target):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    stderr = math.sqrt(acc * (1.0 - acc) / total) if total > 0 else 0.0
    return acc, stderr


def primary_metric_for_task(task_name: str, task_results: dict, task_config: dict) -> str | None:
    """
    Determine the primary accuracy metric name for a task.
    Prefer the first entry in metric_list from the config; fall back to
    scanning the result keys for something that ends with ',none' and has
    no '_stderr' or list value.
    """
    metric_list = task_config.get("metric_list", [])
    if metric_list:
        return metric_list[0]["metric"]

    # Fallback: first scalar metric key ending with ',none'
    for key, val in task_results.items():
        if key.endswith(",none") and "_stderr" not in key and not isinstance(val, list):
            return key.rsplit(",", 1)[0]

    return None


def process_pair(samples_path: Path, results_path: Path, force: bool = False) -> None:
    with open(results_path) as f:
        results = json.load(f)

    modified = False

    for task_name, task_results in results.get("results", {}).items():
        task_config = results.get("configs", {}).get(task_name, {})
        primary = primary_metric_for_task(task_name, task_results, task_config)

        if primary is None:
            print(f"  [SKIP] cannot determine primary metric for task '{task_name}'")
            continue

        acc_key = f"{primary},_extract_answer"
        if acc_key in task_results and not force:
            print(f"  [SKIP] already has '{acc_key}' in {results_path.name}")
            continue

        acc, stderr = compute_accuracy(samples_path)

        task_results[acc_key] = acc
        task_results[f"{primary}_stderr,_extract_answer"] = stderr
        modified = True

        print(
            f"  [{task_name}] {primary}: acc={acc:.4f}  stderr={stderr:.6f}"
            f"  ({samples_path.name})"
        )

    if modified:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> updated {results_path}")


def run(eval_dir: Path, force: bool = False) -> None:
    results_files = sorted(eval_dir.rglob("*_results.json"))

    if not results_files:
        print(f"No *_results.json files found under {eval_dir}")
        return

    for results_path in results_files:
        parent = results_path.parent
        # Timestamp prefix: everything before '_results.json'
        timestamp = results_path.name[: -len("_results.json")]

        samples_files = sorted(parent.glob(f"{timestamp}_samples_*.jsonl"))
        if not samples_files:
            print(f"[SKIP] no samples file for {results_path.relative_to(eval_dir)}")
            continue

        if len(samples_files) > 1:
            print(
                f"[WARN] multiple samples files for {timestamp}, using first: "
                f"{[s.name for s in samples_files]}"
            )

        samples_path = samples_files[0]
        print(f"\n{results_path.relative_to(eval_dir)}")
        process_pair(samples_path, results_path, force=force)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default="outputs/eval_generate_logs",
        help="Root directory to scan (default: outputs/eval_generate_logs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if _extract_answer already present",
    )
    args = parser.parse_args()

    run(Path(args.eval_dir), force=args.force)
