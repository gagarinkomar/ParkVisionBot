import cv2
import numpy as np

from .detect import Detection
from .spots import Spot


def draw_overlay(
    bgr: np.ndarray,
    spots: list[Spot],
    occupied: dict[str, bool],
    detections: list[Detection] | None = None,
) -> np.ndarray:
    img = bgr.copy()

    # Draw spots
    for s in spots:
        pts = np.array(s.polygon, dtype=np.int32).reshape((-1, 1, 2))
        is_occ = bool(occupied.get(s.spot_id, False))
        color = (0, 0, 255) if is_occ else (0, 200, 0)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        # label at polygon centroid
        cx = int(np.mean([p[0] for p in s.polygon]))
        cy = int(np.mean([p[1] for p in s.polygon]))
        cv2.putText(img, s.spot_id, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Draw detections
    if detections:
        for d in detections:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                img,
                f"{d.label} {d.conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

    # Summary
    total = len(spots)
    free = sum(1 for s in spots if not occupied.get(s.spot_id, False))
    cv2.putText(
        img,
        f"FREE {free}/{total}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img
