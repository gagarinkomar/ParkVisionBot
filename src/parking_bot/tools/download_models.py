import argparse
from pathlib import Path

import requests

URLS = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}


def _download(url: str, to_path: Path) -> None:
    to_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    to_path.write_bytes(r.content)


def main() -> None:
    p = argparse.ArgumentParser(description="Download YOLOv4-tiny model files into model dir")
    p.add_argument("--dir", default="/app/data/models", help="Output directory")
    args = p.parse_args()

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"OK: {dst} (already exists)")
            continue
        print(f"Downloading {name}...")
        _download(url, dst)
        print(f"Saved: {dst} ({dst.stat().st_size} bytes)")

    print("Done")


if __name__ == "__main__":
    main()
