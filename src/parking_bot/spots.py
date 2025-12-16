import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Spot:
    spot_id: str
    polygon: list[tuple[int, int]]  # (x, y) points in image pixels


@dataclass(frozen=True)
class SpotsConfig:
    image_size: tuple[int, int] | None  # (w, h)
    spots: list[Spot]


def load_spots(path: str | Path) -> SpotsConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    image_size = data.get("image_size")
    if image_size is not None:
        image_size = (int(image_size[0]), int(image_size[1]))

    spots = []
    for s in data.get("spots", []):
        sid = str(s["id"])
        pts = [(int(x), int(y)) for x, y in s["polygon"]]
        spots.append(Spot(spot_id=sid, polygon=pts))

    return SpotsConfig(image_size=image_size, spots=spots)


def scale_spots(spots_cfg: SpotsConfig, target_size: tuple[int, int]) -> list[Spot]:
    if spots_cfg.image_size is None:
        return spots_cfg.spots

    w0, h0 = spots_cfg.image_size
    w1, h1 = target_size
    if w0 <= 0 or h0 <= 0 or (w0 == w1 and h0 == h1):
        return spots_cfg.spots

    sx = w1 / w0
    sy = h1 / h0

    scaled: list[Spot] = []
    for s in spots_cfg.spots:
        poly = [(int(round(x * sx)), int(round(y * sy))) for x, y in s.polygon]
        scaled.append(Spot(spot_id=s.spot_id, polygon=poly))
    return scaled


def point_in_polygon(point: tuple[float, float], polygon: Iterable[tuple[int, int]]) -> bool:
    """Ray casting algorithm."""
    x, y = point
    poly = list(polygon)
    inside = False
    n = len(poly)
    if n < 3:
        return False

    x0, y0 = poly[-1]
    for x1, y1 in poly:
        intersects = ((y1 > y) != (y0 > y)) and (
            x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-9) + x1
        )
        if intersects:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def spot_occupied(spot: Spot, vehicle_centers: np.ndarray) -> bool:
    """A spot is occupied if at least one detected vehicle center falls inside its polygon."""
    for cx, cy in vehicle_centers:
        if point_in_polygon((float(cx), float(cy)), spot.polygon):
            return True
    return False
