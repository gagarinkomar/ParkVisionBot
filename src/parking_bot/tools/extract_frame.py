import argparse
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser(description="Extract one frame from video")
    p.add_argument("--video", required=True, help="Path to video")
    p.add_argument("--frame", type=int, default=0, help="Frame index")
    p.add_argument("--out", default="frame.png", help="Output image path")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ok, fr = cap.read()
    cap.release()

    if not ok or fr is None:
        raise SystemExit("Cannot read frame")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fr)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
