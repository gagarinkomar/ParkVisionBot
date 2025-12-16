## ParkVisionBot — поиск свободных парковочных мест

Проект по Computer Vision: **парковочные места задаются один раз** (полигонами на кадре), далее детектируется транспорт и вычисляется занятость места (если **центр bbox** попадает внутрь полигона — место занято). Для демонстрации реализован **Telegram-бот**.

### Технологии
- **Python**, **uv**
- **OpenCV**
- **YOLOv4-tiny** (предобученные веса на COCO) — дефолтный детектор
- **YOLOv8 (Ultralytics)** — опционально после fine-tuning
- **Docker / docker compose**
- **python-telegram-bot**

### Структура
- `src/parking_bot/bot.py` — Telegram-бот (картинка → картинка, видео → видео)
- `src/parking_bot/detect.py` — детектор (переключается параметром backend)
- `src/parking_bot/spots.py` — работа с полигонами
- `data/spots.json` — разметка мест
- `data/models/` — `yolov4-tiny.cfg/.weights + coco.names`

---

### Быстрый старт (YOLOv4-tiny, лёгкий Docker)
1) Подготовить `env`:
- `cp env.example env`
- заполнить `TELEGRAM_BOT_TOKEN=...`

2) Скачать модельные файлы:

```bash
docker compose run --rm bot parking-download-models --dir /app/data/models
```

3) Достать кадр для разметки:

```bash
docker compose run --rm bot parking-extract-frame --video /app/video.mp4 --frame 0 --out /app/data/frame0.png
```

4) Разметить парковочные места (через web страницу):

```bash
uv sync
uv run parking-web-mark-spots --image data/frame0.png --out data/spots.json
```

5) проверить пайплайн без Telegram:

```bash
docker compose run --rm bot parking-demo --image /app/data/frame0.png --out /app/data/test.png
```

6) Запустить бота:

```bash
docker compose up --build bot
```

### Как тестировать бота
- **Фото**: отправь картинку → бот вернёт картинку с полигонами и `FREE x/y`
- **Видео**: отправь видео → бот вернёт **аннотированное видео**

Настройки видео в `env`:
- `VIDEO_EVERY=5` (обрабатываем каждый 5-й кадр)
- `VIDEO_MAX_FRAMES=180` (лимит длины, 0 = без лимита)

Если хочется быстро проверить обработку видео без Telegram:

```bash
docker compose run --rm bot parking-demo-video --video /app/video.mp4 --out /app/data/out.mp4 --every 5 --max-frames 120
```

---

### Дообучение через Roboflow + YOLOv8 (опционально)
1) Нарезать кадры:

```bash
docker compose run --rm bot parking-extract-dataset --video /app/video.mp4 --out /app/data/dataset/images_raw --every 15 --max 300
```

2) Roboflow: загрузить кадры, разметить bbox транспорта, экспортировать датасет в формате **YOLOv8** (распаковать в `data/roboflow_dataset/`).

3) Обучить локально:

```bash
uv sync --extra train
uv run parking-train-yolo --data data/roboflow_dataset/data.yaml --model yolov8n.pt --epochs 30 --imgsz 640 --batch 8 --runs-dir data/runs
```

Веса будут в `data/runs/detect/weights/best.pt`.

---

### Запуск бота на YOLOv8
Обычный `bot` специально без PyTorch (лёгкий). Для YOLOv8 есть отдельный сервис `bot_ultra`.

Запуск:

```bash
docker compose up --build bot_ultra
```
