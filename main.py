from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "DATASET" / "data.yaml"
MODEL_WEIGHTS = ROOT / "yolo26n.pt"
PROJECT_DIR = ROOT / "runs"
RUN_NAME = "coal_yolo26n_aug"


TRAIN_ARGS = {
    "data": str(DATA_YAML),
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 2,
    "project": str(PROJECT_DIR),
    "name": RUN_NAME,
    "exist_ok": True,
    "patience": 30,
    "optimizer": "auto",
    "seed": 0,
    "deterministic": True,
    "plots": True,
    # Augmentation tuned for coal/gangue detection: preserve object geometry,
    # vary lighting/scale/position, and use moderate sample mixing.
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.1,
    "close_mosaic": 10,
    "auto_augment": "randaugment",
    "erasing": 0.2,
}


METRIC_COLUMNS = {
    "epoch": "Epoch",
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "train/box_loss": "Train Box Loss",
    "train/cls_loss": "Train Cls Loss",
    "val/box_loss": "Val Box Loss",
    "val/cls_loss": "Val Cls Loss",
}


def _clean_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items()}


def _read_results(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [_clean_row(row) for row in reader]


def _format_value(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value


def _make_metrics_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No metrics were found."

    last_row = rows[-1]
    best_map_row = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    best_map50_row = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))

    table_rows = [
        ("Last epoch", last_row),
        ("Best mAP50-95", best_map_row),
        ("Best mAP50", best_map50_row),
    ]

    headers = ["Checkpoint"] + list(METRIC_COLUMNS.values())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for label, row in table_rows:
        values = [label]
        values.extend(_format_value(row[column]) for column in METRIC_COLUMNS)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _write_summary(save_dir: Path, table: str) -> None:
    summary_path = save_dir / "metrics_summary.md"
    summary = (
        "# YOLO26n Training Metrics\n\n"
        f"- Model: `{MODEL_WEIGHTS.name}`\n"
        f"- Dataset: `{DATA_YAML}`\n"
        f"- Run directory: `{save_dir}`\n\n"
        f"{table}\n"
    )
    summary_path.write_text(summary, encoding="utf-8")


def main() -> None:
    from ultralytics import YOLO

    model = YOLO(str(MODEL_WEIGHTS))
    results = model.train(**TRAIN_ARGS)

    save_dir = Path(results.save_dir)
    results_csv = save_dir / "results.csv"
    rows = _read_results(results_csv)
    table = _make_metrics_table(rows)
    _write_summary(save_dir, table)

    print("\nTraining finished. Performance metrics:\n")
    print(table)
    print(f"\nMetrics summary saved to: {save_dir / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()