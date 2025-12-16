import argparse
from pathlib import Path

import cv2

from ..config import load_settings
from ..detect import VehicleDetector, centers_from_detections
from ..spots import load_spots, scale_spots, spot_occupied
from ..viz import draw_overlay


def _make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer mp4
    if path.suffix.lower() == ".mp4":
        for fourcc in ("mp4v", "avc1"):
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if vw.isOpened():
                return vw
    # Fallback
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not vw.isOpened():
        raise SystemExit(
            "Cannot open VideoWriter. Try output .avi (XVID) or ensure codecs are available in your OpenCV build."
        )
    return vw


def main() -> None:
    p = argparse.ArgumentParser(description="Offline demo: video -> annotated video + occupancy")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out", default="out.mp4", help="Output annotated video")
    p.add_argument("--every", type=int, default=1, help="Process every N-th frame")
    p.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    p.add_argument("--no-dets", action="store_true", help="Do not draw detection bboxes (only spots)")
    args = p.parse_args()

    settings = load_settings()
    spots_cfg = load_spots(settings.spots_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    det = VehicleDetector(
        backend=settings.detector_backend,
        model_dir=settings.model_dir,
        cfg_name=settings.yolo_cfg,
        weights_name=settings.yolo_weights,
        coco_names_name=settings.coco_names,
        ultralytics_model=settings.ultralytics_model,
        conf_thres=settings.conf_thres,
    )

    spots = scale_spots(spots_cfg, (w, h))

    out_path = Path(args.out)
    writer = _make_writer(out_path, fps, (w, h))

    idx = 0
    written = 0
    last_occ = {s.spot_id: False for s in spots}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if args.every > 1 and (idx % args.every != 0):
            idx += 1
            continue

        dets = det.detect(frame)
        centers = centers_from_detections(dets)
        occ = {s.spot_id: spot_occupied(s, centers) for s in spots}
        last_occ = occ

        overlay = draw_overlay(frame, spots, occ, detections=None if args.no_dets else dets)
        writer.write(overlay)
        written += 1

        idx += 1
        if args.max_frames and written >= args.max_frames:
            break

    cap.release()
    writer.release()

    total = len(spots)
    free = sum(1 for s in spots if not last_occ.get(s.spot_id, False))
    print(f"Saved: {out_path} | last FREE {free}/{total}")


if __name__ == "__main__":
    main()
