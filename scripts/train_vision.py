"""
Rebuild the vision dataset with a reproducible split, retrain YOLOv8s-cls, and
evaluate on a held-out TEST split the training never touched.

Why retrain instead of just evaluating the existing best.pt? Its split was made
with an unseeded shuffle, so the original test images can't be reconstructed —
any "test" set rebuilt now may overlap its training images. Retraining on a
deterministic split (sorted file names + per-class seeded shuffle) gives a test
score that can be defended.

Outputs (in --out):
    weights/best.pt           trained classifier
    test_metrics.json         accuracy, macro-F1, per-class P/R/F1, ECE, coverage at the
                              confidence gates used by src/tools/vision_tool.py
    confusion_matrix.csv

Usage (Kaggle GPU):
    python -m scripts.train_vision --plantvillage <dir> --pests <dir> --out outputs/farm-vision-v2
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_PER_CLASS = 600
TRAIN_R, VAL_R = 0.80, 0.10          # test = remaining 10%
SEED = 42


def find_class_dirs(root: Path) -> list[Path]:
    """Directories that directly contain images (handles nested Kaggle layouts)."""
    dirs = {p.parent for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    return sorted(dirs)


def build_split(class_dirs: dict[str, Path], dataset_dir: Path) -> dict:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    counts = {}
    for name, src in sorted(class_dirs.items()):
        images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        random.Random(f"{SEED}-{name}").shuffle(images)
        images = images[:MAX_PER_CLASS]
        n_train, n_val = int(len(images) * TRAIN_R), int(len(images) * VAL_R)
        parts = {"train": images[:n_train], "val": images[n_train:n_train + n_val], "test": images[n_train + n_val:]}
        for split, files in parts.items():
            dest = dataset_dir / split / name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
        counts[name] = {k: len(v) for k, v in parts.items()}
    return counts


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate(model, dataset_dir: Path, out_dir: Path) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    names = [model.names[i] for i in range(len(model.names))]
    paths, y_true = [], []
    for idx, name in enumerate(names):
        for p in sorted((dataset_dir / "test" / name).iterdir()):
            paths.append(str(p))
            y_true.append(idx)

    y_pred, conf = [], []
    for start in range(0, len(paths), 64):
        for r in model(paths[start:start + 64], verbose=False):
            y_pred.append(int(r.probs.top1))
            conf.append(float(r.probs.top1conf))
    y_true, y_pred, conf = np.array(y_true), np.array(y_pred), np.array(conf)
    correct = (y_true == y_pred).astype(float)

    report = classification_report(y_true, y_pred, labels=list(range(len(names))),
                                   target_names=names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(names), comments="")

    gates = {}
    for threshold in (0.40, 0.70):
        kept = conf >= threshold
        gates[f"conf>={threshold}"] = {
            "coverage": round(float(kept.mean()), 4),
            "accuracy_on_kept": round(float(correct[kept].mean()), 4) if kept.any() else None,
        }

    worst = sorted(((n, report[n]["f1-score"]) for n in names), key=lambda kv: kv[1])[:5]
    return {
        "n_test_images": int(len(y_true)),
        "top1_accuracy": round(float(correct.mean()), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "ece_10_bins": round(expected_calibration_error(conf, correct), 4),
        "confidence_gates": gates,
        "worst_classes_by_f1": [{"class": n, "f1": round(f, 4)} for n, f in worst],
        "per_class": {n: {k: round(report[n][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                      for n in names},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantvillage", required=True, help="folder containing the PlantVillage class folders")
    ap.add_argument("--pests", required=True, help="folder containing the agricultural pest class folders")
    ap.add_argument("--work", default="/kaggle/working/farm-vision-dataset")
    ap.add_argument("--out", default="outputs/farm-vision-v2")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = {d.name: d for d in find_class_dirs(Path(args.plantvillage))}
    class_dirs.update({f"pest_{d.name.lower().replace(' ', '_')}": d for d in find_class_dirs(Path(args.pests))})
    counts = build_split(class_dirs, Path(args.work))
    totals = {s: sum(c[s] for c in counts.values()) for s in ("train", "val", "test")}
    print(f"{len(counts)} classes | {totals}")

    from ultralytics import YOLO

    model = YOLO("yolov8s-cls.pt")
    model.train(data=args.work, epochs=args.epochs, imgsz=224, batch=64, patience=10,
                optimizer="AdamW", lr0=0.001, seed=0, deterministic=True,
                project=str(out_dir.resolve()), name="train", exist_ok=True)
    # Ultralytics >= 8.4 nests a relative `project` under runs/<task>/, so ask the
    # trainer where it actually wrote the weights instead of assuming the path.
    best = Path(model.trainer.best)
    (out_dir / "weights").mkdir(exist_ok=True)
    shutil.copy2(best, out_dir / "weights" / "best.pt")

    metrics = evaluate(YOLO(str(out_dir / "weights" / "best.pt")), Path(args.work), out_dir)
    metrics["split_counts"] = totals
    metrics["n_classes"] = len(counts)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
