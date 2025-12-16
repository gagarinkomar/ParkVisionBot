import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Train (fine-tune) YOLO on your parking dataset")
    p.add_argument("--data", required=True, help="Path to dataset YAML (Ultralytics format)")
    p.add_argument("--model", default="yolov8n.pt", help="Base model")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--runs-dir",
        default="data/runs",
        help="Where to save training runs (default: data/runs). Useful if ./runs is not writable.",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(
            "Ultralytics is not installed.\n"
            "Install training extra and retry:\n"
            "  uv sync --extra train\n"
        ) from e

    model = YOLO(args.model)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=2,
        device="cpu",
        project=str(runs_dir),
        name="detect",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
