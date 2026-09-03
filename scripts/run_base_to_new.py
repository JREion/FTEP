"""
Resumable, visible base-to-new script for fine-tuning FTEP and host models.
For testing, use ``--dry-run`` to print the complete command sequence without launching anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Optional


DATASET_INFO = {
    "imagenet": ("ImageNet", "imagenet-1k", "imagenet.yaml"),
    "caltech101": ("Caltech101", "caltech-101", "caltech101.yaml"),
    "oxford_pets": ("OxfordPets", "oxford_pets", "oxford_pets.yaml"),
    "stanford_cars": ("StanfordCars", "stanford_cars", "stanford_cars.yaml"),
    "flowers102": ("OxfordFlowers", "flowers-102", "oxford_flowers.yaml"),
    "food101": ("Food101", "food101", "food101.yaml"),
    "fgvc_aircraft": ("FGVCAircraft", "fgvc_aircraft/fgvc-aircraft-2013b", "fgvc_aircraft.yaml"),
    "sun397": ("SUN397", "sun-397", "sun397.yaml"),
    "dtd": ("DescribableTextures", "dtd/dtd", "dtd.yaml"),
    "eurosat": ("EuroSAT", "eurosat", "eurosat.yaml"),
    "ucf101": ("UCF101", "ucf101", "ucf101.yaml"),
}

METHOD_INFO = {
    "CoOp": {
        "trainer": "CoOp",
        "config": "configs/trainers/CoOp/vit_b16_ep10.yaml",
        "model_names": ("prompt_learner",),
    },
    "DePT": {
        "trainer": "DePT",
        "config": "configs/trainers/DePT/vit_b16_c2_ep10_batch4_4+4ctx_lr35.yaml",
        "model_names": ("dept_model",),
    },
    "MMRL": {
        "trainer": "MMRL",
        "config": "configs/trainers/MMRL/vit_b16.yaml",
        "model_names": ("MultiModalRepresentationLearner",),
    },
    "FTEP_MMRL": {
        "trainer": "FTEP_MMRL",
        "config": "configs/trainers/FTEP_MMRL/vit_b16.yaml",
        "model_names": ("MultiModalRepresentationLearner",),
    },
}


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _command_text(command: List[str]) -> str:
    # list2cmdline produces a directly pasteable Windows command line.
    return subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)


def resolve_checkpoint(output_dir: Path, model_names: Iterable[str]) -> Optional[Path]:
    """Return the newest readable checkpoint for any registered model name."""
    candidates = []
    for model_name in model_names:
        model_dir = output_dir / model_name
        if not model_dir.is_dir():
            continue
        for path in model_dir.iterdir():
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            if path.name == "model-best.pth.tar":
                candidates.append((10**12, path))
            elif path.name.startswith("model.pth.tar-"):
                suffix = path.name.rsplit("-", 1)[-1]
                if suffix.isdigit():
                    candidates.append((int(suffix), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime))[1]


def checkpoint_readable(path: Path) -> bool:
    """Validate that a checkpoint can be deserialized by the active PyTorch."""
    try:
        import torch
        torch.load(str(path), map_location="cpu")
        return True
    except Exception as exc:
        print(f"[FTEP] checkpoint unreadable: {path}: {exc}")
        return False


def _write_manifest(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_command(
    command: List[str],
    *,
    cwd: Path,
    record_base: Dict,
    manifest: Path,
    dry_run: bool,
) -> int:
    command_text = _command_text(command)
    record = dict(record_base)
    record.update({"started_at": _timestamp(), "command": command_text})
    print(f"\n[FTEP] {record['method']} / {record['dataset']} / seed {record['seed']} / {record['split']} / {record['operation']}")
    print(f"COMMAND: {command_text}", flush=True)
    if dry_run:
        record.update({"ended_at": _timestamp(), "return_code": 0, "status": "dry_run"})
        _write_manifest(manifest, record)
        return 0

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    record.update({
        "ended_at": _timestamp(),
        "return_code": return_code,
        "status": "completed" if return_code == 0 else "failed",
    })
    _write_manifest(manifest, record)
    return return_code


def _paths(repo_root: Path, output_root: str, data_root: str):
    out = Path(output_root)
    data = Path(data_root)
    if not out.is_absolute():
        out = repo_root / out
    if not data.is_absolute():
        data = repo_root / data
    return out, data


def _train_command(repo_root: Path, method: str, dataset: str, seed: int, shots: int, epochs: int, output_dir: Path, data_root: Path) -> List[str]:
    display_name, relative_data, dataset_config = DATASET_INFO[dataset]
    info = METHOD_INFO[method]
    return [
        sys.executable, "-u", "train.py",
        "--root", str(data_root / relative_data),
        "--seed", str(seed), "--trainer", info["trainer"],
        "--dataset-config-file", str(repo_root / "configs" / "datasets" / dataset_config),
        "--config-file", str(repo_root / info["config"]),
        "--output-dir", str(output_dir),
        "DATASET.NUM_SHOTS", str(shots),
        "DATASET.SUBSAMPLE_CLASSES", "base",
        "TASK", "B2N",
        "OPTIM.MAX_EPOCH", str(epochs),
    ]


def _eval_command(repo_root: Path, method: str, dataset: str, seed: int, shots: int, epochs: int, split: str, output_dir: Path, base_dir: Path, data_root: Path) -> List[str]:
    _, relative_data, dataset_config = DATASET_INFO[dataset]
    info = METHOD_INFO[method]
    command = [
        sys.executable, "-u", "train.py",
        "--root", str(data_root / relative_data),
        "--seed", str(seed), "--trainer", info["trainer"],
        "--dataset-config-file", str(repo_root / "configs" / "datasets" / dataset_config),
        "--config-file", str(repo_root / info["config"]),
        "--output-dir", str(output_dir), "--model-dir", str(base_dir),
        "--load-epoch", "-1", "--eval-only",
        "DATASET.NUM_SHOTS", str(shots),
        "DATASET.SUBSAMPLE_CLASSES", split,
        "TASK", "B2N",
        "OPTIM.MAX_EPOCH", str(epochs),
    ]
    return command


def _marker(output_dir: Path, name: str) -> Path:
    return output_dir / f".{name}.complete.json"


def _run_stage(args, repo_root: Path, output_root: Path, data_root: Path, method: str, dataset: str, seed: int, split: str, operation: str, command: List[str], output_dir: Path, checkpoint: Optional[Path] = None) -> int:
    marker = _marker(output_dir, operation)
    if marker.exists() and not args.dry_run:
        print(f"[FTEP] skip completed stage: {marker}")
        return 0
    base = {
        "method": method, "trainer": METHOD_INFO[method]["trainer"],
        "dataset": dataset, "dataset_name": DATASET_INFO[dataset][0],
        "seed": seed, "split": split, "shots": args.shots,
        "epochs": args.epochs, "checkpoint_path": str(checkpoint) if checkpoint else "",
        "output_path": str(output_dir), "operation": operation,
    }
    code = _run_command(command, cwd=repo_root, record_base=base, manifest=output_root / "run_manifest.jsonl", dry_run=args.dry_run)
    if code == 0 and not args.dry_run:
        marker.write_text(json.dumps({"completed_at": _timestamp(), "command": _command_text(command)}, indent=2), encoding="utf-8")
    return code


def run(args) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    output_root, data_root = _paths(repo_root, args.output_root, args.data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.method:
        args.methods = [args.method]
    for method in args.methods:
        if method not in METHOD_INFO:
            raise SystemExit(f"Unsupported method {method}; choose from {', '.join(METHOD_INFO)}")
    for dataset in args.datasets:
        if dataset not in DATASET_INFO:
            raise SystemExit(f"Unsupported dataset {dataset}; choose from {', '.join(DATASET_INFO)}")

    for method in args.methods:
        for dataset in args.datasets:
            for seed in args.seeds:
                base_dir = output_root / method / dataset / "base" / f"seed{seed}"
                new_dir = output_root / method / dataset / "new" / f"seed{seed}"
                if args.phase == "base":
                    train_code = _run_stage(
                        args, repo_root, output_root, data_root, method, dataset, seed,
                        "base", "train", _train_command(repo_root, method, dataset, seed, args.shots, args.epochs, base_dir, data_root), base_dir,
                    )
                    checkpoint = Path("dry-run-checkpoint") if args.dry_run else resolve_checkpoint(base_dir, METHOD_INFO[method]["model_names"])
                    if train_code != 0:
                        continue
                    if checkpoint is None:
                        record = {
                            "method": method, "trainer": METHOD_INFO[method]["trainer"], "dataset": dataset,
                            "dataset_name": DATASET_INFO[dataset][0], "seed": seed, "split": "base",
                            "shots": args.shots, "epochs": args.epochs, "checkpoint_path": "",
                            "output_path": str(base_dir), "operation": "checkpoint", "status": "missing_base_checkpoint",
                            "started_at": _timestamp(), "ended_at": _timestamp(), "return_code": None,
                            "command": "", "error": "base checkpoint was not found after training",
                        }
                        _write_manifest(output_root / "run_manifest.jsonl", record)
                        print(f"[FTEP] missing_base_checkpoint: {base_dir}")
                        continue
                    if not args.dry_run and not checkpoint_readable(checkpoint):
                        record = {
                            "method": method, "trainer": METHOD_INFO[method]["trainer"], "dataset": dataset,
                            "dataset_name": DATASET_INFO[dataset][0], "seed": seed, "split": "base",
                            "shots": args.shots, "epochs": args.epochs, "checkpoint_path": str(checkpoint),
                            "output_path": str(base_dir), "operation": "checkpoint", "status": "unreadable_base_checkpoint",
                            "started_at": _timestamp(), "ended_at": _timestamp(), "return_code": None,
                            "command": "", "error": "checkpoint could not be deserialized by torch.load",
                        }
                        _write_manifest(output_root / "run_manifest.jsonl", record)
                        continue
                    base_code = _run_stage(
                        args, repo_root, output_root, data_root, method, dataset, seed,
                        "base", "eval", _eval_command(repo_root, method, dataset, seed, args.shots, args.epochs, "base", base_dir, base_dir, data_root), base_dir, checkpoint,
                    )
                    if base_code != 0:
                        continue
                else:
                    checkpoint = Path("dry-run-checkpoint") if args.dry_run else resolve_checkpoint(base_dir, METHOD_INFO[method]["model_names"])
                    if checkpoint is None:
                        record = {
                            "method": method, "trainer": METHOD_INFO[method]["trainer"], "dataset": dataset,
                            "dataset_name": DATASET_INFO[dataset][0], "seed": seed, "split": "new",
                            "shots": args.shots, "epochs": args.epochs, "checkpoint_path": "",
                            "output_path": str(new_dir), "operation": "checkpoint", "status": "missing_base_checkpoint",
                            "started_at": _timestamp(), "ended_at": _timestamp(), "return_code": None,
                            "command": "", "error": "new evaluation requires an existing base checkpoint",
                        }
                        _write_manifest(output_root / "run_manifest.jsonl", record)
                        print(f"[FTEP] missing_base_checkpoint: {base_dir}")
                        continue
                    if not args.dry_run and not checkpoint_readable(checkpoint):
                        record = {
                            "method": method, "trainer": METHOD_INFO[method]["trainer"], "dataset": dataset,
                            "dataset_name": DATASET_INFO[dataset][0], "seed": seed, "split": "new",
                            "shots": args.shots, "epochs": args.epochs, "checkpoint_path": str(checkpoint),
                            "output_path": str(new_dir), "operation": "checkpoint", "status": "unreadable_base_checkpoint",
                            "started_at": _timestamp(), "ended_at": _timestamp(), "return_code": None,
                            "command": "", "error": "checkpoint could not be deserialized by torch.load",
                        }
                        _write_manifest(output_root / "run_manifest.jsonl", record)
                        continue
                    base_code = _run_stage(
                        args, repo_root, output_root, data_root, method, dataset, seed,
                        "base", "eval", _eval_command(repo_root, method, dataset, seed, args.shots, args.epochs, "base", base_dir, base_dir, data_root), base_dir, checkpoint,
                    )
                    if base_code == 0:
                        _run_stage(
                            args, repo_root, output_root, data_root, method, dataset, seed,
                            "new", "eval", _eval_command(repo_root, method, dataset, seed, args.shots, args.epochs, "new", new_dir, base_dir, data_root), new_dir, checkpoint,
                        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["CoOp"], choices=tuple(METHOD_INFO))
    parser.add_argument("--method", choices=tuple(METHOD_INFO), default=None, help="single-method alias for --methods")
    parser.add_argument("--datasets", nargs="+", required=True, choices=tuple(DATASET_INFO))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--phase", choices=("base", "new"), default="base")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--data-root", default="DATA")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
