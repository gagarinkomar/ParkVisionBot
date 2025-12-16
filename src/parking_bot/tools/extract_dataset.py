import argparse
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frames from video for manual labeling")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out", default="data/dataset/images", help="Output directory")
    p.add_argument("--every", type=int, default=15, help="Save every N-th frame")
    p.add_argument("--max", type=int, default=500, help="Max frames to save")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    saved = 0
    idx = 0
    while saved < args.max:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if idx % args.every == 0:
            path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), fr)
            saved += 1
        idx += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
