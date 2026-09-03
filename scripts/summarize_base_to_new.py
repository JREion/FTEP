"""Summarize visible runner manifests and Dassl logs into CSV/Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Dict, List


FIELDS = [
    "method", "backbone", "dataset", "seed", "split", "shots", "epochs",
    "Base accuracy", "New accuracy", "HM", "checkpoint path", "output path",
    "exact command", "error message",
]


def _accuracy(path: Path):
    if not path:
        return None
    candidates = []
    for log in [path / "log.txt", *path.glob("log-*.txt")]:
        if not log.is_file():
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        candidates.extend(float(value) for value in re.findall(r"\* accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", text))
    return candidates[-1] if candidates else None


def _hm(base, new):
    if base is None or new is None or base + new == 0:
        return None
    return 2 * base * new / (base + new)


def _load_records(manifest: Path) -> List[Dict]:
    if not manifest.is_file():
        return []
    records = []
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarize(output_root: Path, results_root: Path) -> Path:
    records = _load_records(output_root / "run_manifest.jsonl")
    by_key = {}
    for record in records:
        if record.get("operation") not in {"eval", "checkpoint"}:
            continue
        key = (record.get("method"), record.get("dataset"), record.get("seed"), record.get("split"))
        by_key[key] = record

    rows = []
    grouped = {}
    for (method, dataset, seed, split), record in sorted(by_key.items(), key=str):
        out = Path(record.get("output_path", ""))
        base_dir = output_root / method / dataset / "base" / f"seed{seed}"
        new_dir = output_root / method / dataset / "new" / f"seed{seed}"
        base_acc = _accuracy(base_dir)
        new_acc = _accuracy(new_dir)
        status = record.get("status", "")
        error = record.get("error", "")
        if status == "missing_base_checkpoint":
            error = error or "missing_base_checkpoint"
        row = {
            "method": method or "", "backbone": "ViT-B/16", "dataset": dataset or "",
            "seed": seed if seed is not None else "", "split": split or "",
            "shots": record.get("shots", 16), "epochs": record.get("epochs", 10),
            "Base accuracy": base_acc if split == "base" else None,
            "New accuracy": new_acc if split == "new" else None,
            "HM": _hm(base_acc, new_acc) if split == "new" else None,
            "checkpoint path": record.get("checkpoint_path", ""),
            "output path": str(out), "exact command": record.get("command", ""),
            "error message": error,
        }
        rows.append(row)
        grouped.setdefault((method, dataset, seed), {})[split] = row

    # Add a combined base/new row when both evaluations exist.
    combined = []
    for (method, dataset, seed), parts in sorted(grouped.items(), key=str):
        if "base" not in parts or "new" not in parts:
            continue
        base = parts["base"].get("Base accuracy") or _accuracy(output_root / method / dataset / "base" / f"seed{seed}")
        new = parts["new"].get("New accuracy") or _accuracy(output_root / method / dataset / "new" / f"seed{seed}")
        combined.append({
            "method": method, "backbone": "ViT-B/16", "dataset": dataset, "seed": seed,
            "split": "base+new", "shots": parts["base"]["shots"], "epochs": parts["base"]["epochs"],
            "Base accuracy": base, "New accuracy": new, "HM": _hm(base, new),
            "checkpoint path": parts["new"].get("checkpoint path") or parts["base"].get("checkpoint path"),
            "output path": parts["new"].get("output path", ""),
            "exact command": parts["new"].get("exact command", ""),
            "error message": parts["base"].get("error message") or parts["new"].get("error message", ""),
        })
    rows.extend(combined)

    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "base_to_new_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = results_root / "report.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Base-to-new results\n\n")
        handle.write("Missing or failed runs are retained as explicit error rows; no values are inferred.\n\n")
        handle.write("| Method | Dataset | Seed | Split | Base | New | HM | Error |\n|---|---|---:|---|---:|---:|---:|---|\n")
        for row in rows:
            def fmt(value):
                return "missing" if value in (None, "") else f"{value:.2f}" if isinstance(value, float) else str(value)
            handle.write(f"| {row['method']} | {row['dataset']} | {row['seed']} | {row['split']} | {fmt(row['Base accuracy'])} | {fmt(row['New accuracy'])} | {fmt(row['HM'])} | {row['error message']} |\n")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    results = Path(args.results_root)
    if not results.is_absolute():
        results = Path(__file__).resolve().parents[1] / results
    print(f"Wrote {summarize(root, results)}")
