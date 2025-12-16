FROM python:3.11-slim

# Minimal OS deps for OpenCV (headless)
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv for reproducible installs
RUN pip install --no-cache-dir uv

# Install deps first (better layer cache)
COPY pyproject.toml /app/pyproject.toml
# uv.lock will be generated locally; if absent, we still allow sync.
COPY uv.lock /app/uv.lock
ARG UV_SYNC_EXTRAS="--extra headless"
RUN uv sync --frozen --no-dev ${UV_SYNC_EXTRAS} --no-install-project || uv sync --no-dev ${UV_SYNC_EXTRAS} --no-install-project

COPY src/ /app/src/
COPY data/ /app/data/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Install the project itself (entrypoints), but don't reinstall deps
RUN uv pip install -e . --no-deps

CMD ["parking-bot"]
