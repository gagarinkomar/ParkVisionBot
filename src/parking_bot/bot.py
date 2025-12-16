import tempfile
from pathlib import Path

import cv2
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .config import load_settings
from .detect import VehicleDetector, centers_from_detections
from .spots import load_spots, scale_spots, spot_occupied
from .viz import draw_overlay


def _analyze_bgr(detector: VehicleDetector, spots_path: str, bgr):
    spots_cfg = load_spots(spots_path)
    dets = detector.detect(bgr)
    centers = centers_from_detections(dets)
    spots = scale_spots(spots_cfg, (bgr.shape[1], bgr.shape[0]))
    occ = {s.spot_id: spot_occupied(s, centers) for s in spots}
    overlay = draw_overlay(bgr, spots, occ, detections=dets)
    total = len(spots)
    free = sum(1 for s in spots if not occ.get(s.spot_id, False))
    return overlay, free, total


def _make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    w, h = size
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp4":
        for fourcc in ("mp4v", "avc1"):
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if vw.isOpened():
                return vw
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("Cannot open VideoWriter")
    return vw


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Пришли фото парковки — я верну разметку и количество свободных мест.\n"
        "Важно: сначала надо задать паркоместа в data/spots.json (полигоны)."
    )


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    detector: VehicleDetector = context.application.bot_data["detector"]

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    photo = update.message.photo[-1]
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.jpg"
        out_path = Path(td) / "out.jpg"

        file = await photo.get_file()
        await file.download_to_drive(str(in_path))

        bgr = cv2.imread(str(in_path))
        if bgr is None:
            await update.message.reply_text("Не смог прочитать изображение")
            return

        overlay, free, total = _analyze_bgr(detector, settings.spots_path, bgr)
        cv2.imwrite(str(out_path), overlay)

        caption = f"Свободно: {free}/{total}"
        await update.message.reply_photo(photo=open(out_path, "rb"), caption=caption)


async def on_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.application.bot_data["settings"]
    detector: VehicleDetector = context.application.bot_data["detector"]

    await update.message.chat.send_action(ChatAction.UPLOAD_VIDEO)

    msg = update.message
    if msg is None:
        return
    vid = msg.video
    doc = msg.document
    if vid is None and doc is None:
        return

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.mp4"
        out_path = Path(td) / "out.mp4"

        file = await (vid.get_file() if vid is not None else doc.get_file())
        await file.download_to_drive(str(in_path))

        spots_cfg = load_spots(settings.spots_path)
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            await update.message.reply_text("Не смог прочитать видео")
            return

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        every = int(settings.video_every)
        fps_out = max(1.0, fps_in / max(1, every))

        spots = scale_spots(spots_cfg, (w, h))
        writer = _make_writer(out_path, fps_out, (w, h))

        idx = 0
        written = 0
        last_free = 0
        total = len(spots)

        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if every > 1 and (idx % every != 0):
                idx += 1
                continue

            dets = detector.detect(fr)
            centers = centers_from_detections(dets)
            occ = {s.spot_id: spot_occupied(s, centers) for s in spots}
            last_free = sum(1 for s in spots if not occ.get(s.spot_id, False))

            overlay = draw_overlay(fr, spots, occ, detections=dets)
            writer.write(overlay)
            written += 1
            idx += 1

            if settings.video_max_frames and written >= int(settings.video_max_frames):
                break

        cap.release()
        writer.release()

        caption = f"Свободно (последний кадр): {last_free}/{total}"
        try:
            await update.message.reply_video(video=open(out_path, "rb"), caption=caption)
        except Exception:
            await update.message.reply_document(document=open(out_path, "rb"), caption=caption)


def main() -> None:
    settings = load_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN. Put it into ./env and run via docker compose.")

    # One detector instance for the whole bot
    detector = VehicleDetector(
        backend=settings.detector_backend,
        model_dir=settings.model_dir,
        cfg_name=settings.yolo_cfg,
        weights_name=settings.yolo_weights,
        coco_names_name=settings.coco_names,
        ultralytics_model=settings.ultralytics_model,
        conf_thres=settings.conf_thres,
    )

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["detector"] = detector

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO | filters.Document.MimeType("video/mp4"), on_video))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
