import argparse
from pathlib import Path

import cv2

from .config import load_settings
from .detect import VehicleDetector, centers_from_detections
from .spots import load_spots, scale_spots, spot_occupied
from .viz import draw_overlay


def main() -> None:
    p = argparse.ArgumentParser(description="Offline demo: image -> parking occupancy")
    p.add_argument("--image", required=True, help="Path to image (jpg/png)")
    p.add_argument("--out", default="out.png", help="Output image path")
    args = p.parse_args()

    settings = load_settings()
    spots_cfg = load_spots(settings.spots_path)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    det = VehicleDetector(
        backend=settings.detector_backend,
        model_dir=settings.model_dir,
        cfg_name=settings.yolo_cfg,
        weights_name=settings.yolo_weights,
        coco_names_name=settings.coco_names,
        ultralytics_model=settings.ultralytics_model,
        conf_thres=settings.conf_thres,
    )
    dets = det.detect(img)
    centers = centers_from_detections(dets)

    spots = scale_spots(spots_cfg, (img.shape[1], img.shape[0]))
    occ = {s.spot_id: spot_occupied(s, centers) for s in spots}
    out = draw_overlay(img, spots, occ, detections=dets)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
