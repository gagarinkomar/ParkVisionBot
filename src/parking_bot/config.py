import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str | None
    spots_path: str
    detector_backend: str
    model_dir: str
    yolo_cfg: str
    yolo_weights: str
    coco_names: str
    ultralytics_model: str
    conf_thres: float
    video_every: int
    video_max_frames: int


def load_settings() -> Settings:
    token = _env("TELEGRAM_BOT_TOKEN")

    spots_path = _env("PARKING_SPOTS_PATH", "/app/data/spots.json")
    detector_backend = _env("DETECTOR_BACKEND", "opencv")
    model_dir = _env("MODEL_DIR", "/app/data/models")
    yolo_cfg = _env("YOLO_CFG", "yolov4-tiny.cfg")
    yolo_weights = _env("YOLO_WEIGHTS", "yolov4-tiny.weights")
    coco_names = _env("COCO_NAMES", "coco.names")
    ultralytics_model = _env("ULTRALYTICS_MODEL", "yolov8n.pt")
    conf = float(_env("CONF_THRES", "0.25"))
    video_every = int(_env("VIDEO_EVERY", "5"))
    video_max_frames = int(_env("VIDEO_MAX_FRAMES", "180"))

    return Settings(
        telegram_bot_token=token,
        spots_path=spots_path,
        detector_backend=detector_backend,
        model_dir=model_dir,
        yolo_cfg=yolo_cfg,
        yolo_weights=yolo_weights,
        coco_names=coco_names,
        ultralytics_model=ultralytics_model,
        conf_thres=conf,
        video_every=max(1, video_every),
        video_max_frames=max(0, video_max_frames),
    )
