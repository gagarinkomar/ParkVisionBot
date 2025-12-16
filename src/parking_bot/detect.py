from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


_VEHICLE_LABELS_CANON = {"car", "motorcycle", "bus", "truck"}


def _canon_label(label: str) -> str:
    if label == "motorbike":
        return "motorcycle"
    return label


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    label: str


class VehicleDetector:
    def __init__(
        self,
        backend: str,
        model_dir: str | None = None,
        cfg_name: str = "yolov4-tiny.cfg",
        weights_name: str = "yolov4-tiny.weights",
        coco_names_name: str = "coco.names",
        ultralytics_model: str = "yolov8n.pt",
        conf_thres: float = 0.25,
        nms_thres: float = 0.4,
        input_size: int = 416,
    ):
        self.backend = backend.strip().lower()
        self.conf_thres = float(conf_thres)
        self.nms_thres = float(nms_thres)
        self.input_size = int(input_size)

        if self.backend not in {"opencv", "ultralytics"}:
            raise ValueError("backend must be 'opencv' or 'ultralytics'")

        if self.backend == "opencv":
            if not model_dir:
                raise ValueError("model_dir is required for backend='opencv'")
            self.model_dir = Path(model_dir)
            self.cfg_path = self.model_dir / cfg_name
            self.weights_path = self.model_dir / weights_name
            self.names_path = self.model_dir / coco_names_name

            if not self.cfg_path.exists() or not self.weights_path.exists() or not self.names_path.exists():
                raise RuntimeError(
                    "YOLO model files not found. Run `parking-download-models` to download into data/models. "
                    f"Expected: {self.cfg_path}, {self.weights_path}, {self.names_path}"
                )

            self.class_names = [
                x.strip() for x in self.names_path.read_text(encoding="utf-8").splitlines() if x.strip()
            ]
            self.net = cv2.dnn.readNetFromDarknet(str(self.cfg_path), str(self.weights_path))
            layer_names = self.net.getLayerNames()
            out_layers = self.net.getUnconnectedOutLayers()
            self.out_layer_names = [layer_names[i - 1] for i in out_layers.flatten()]
        else:
            try:
                from ultralytics import YOLO
            except Exception as e:
                raise RuntimeError(
                    "Ultralytics backend requested, but ultralytics is not installed.\n"
                    "Install:\n"
                    "  uv sync --extra train\n"
                ) from e
            self.ultra = YOLO(ultralytics_model)

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        if self.backend == "ultralytics":
            res = self.ultra.predict(bgr_image, conf=self.conf_thres, verbose=False)[0]
            out: list[Detection] = []
            if res.boxes is None:
                return out

            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names

            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                label = _canon_label(names.get(int(k), str(int(k))))
                if label not in _VEHICLE_LABELS_CANON:
                    continue
                out.append(Detection(xyxy=(float(x1), float(y1), float(x2), float(y2)), conf=float(c), label=label))
            return out

        h, w = bgr_image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            bgr_image,
            1 / 255.0,
            (self.input_size, self.input_size),
            (0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layer_names)

        boxes_xywh: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for out in outs:
            for det in out:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf < self.conf_thres:
                    continue
                raw_label = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else str(class_id)
                label = _canon_label(raw_label)
                if label not in _VEHICLE_LABELS_CANON:
                    continue

                cx = int(det[0] * w)
                cy = int(det[1] * h)
                bw = int(det[2] * w)
                bh = int(det[3] * h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes_xywh.append([x, y, bw, bh])
                confidences.append(conf)
                class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_thres, self.nms_thres)
        out: list[Detection] = []
        if len(idxs) == 0:
            return out

        for i in idxs.flatten().tolist():
            x, y, bw, bh = boxes_xywh[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            cid = class_ids[i]
            raw_label = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            out.append(
                Detection(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    conf=float(confidences[i]),
                    label=_canon_label(raw_label),
                )
            )
        return out


def centers_from_detections(dets: list[Detection]) -> np.ndarray:
    if not dets:
        return np.zeros((0, 2), dtype=np.float32)
    centers = []
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return np.array(centers, dtype=np.float32)
