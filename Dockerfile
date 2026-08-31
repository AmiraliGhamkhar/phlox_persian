# syntax=docker/dockerfile:1
#
# Phlox - production image (FastAPI backend + prebuilt React frontend).
#
# Build:  docker build -t localhost/phlox:latest .
# Run:    docker compose up -d            (see docker-compose.yml, reads .env)
#
# The image listens on :5000 and serves both the API and the built SPA, so a
# single published port is all that is needed.
#
# BuildKit cache mounts (--mount=type=cache) keep the npm, apt and uv download
# caches warm across rebuilds even when a layer is invalidated; they never end
# up in the shipped image. Requires BuildKit (default in Docker 23+ / buildx).

###############################################################################
# Stage 1 - build the React app
###############################################################################
FROM node:24-slim AS build

WORKDIR /usr/src/app

# Install dependencies first so the layer is cached unless the lockfile or npm
# policy files change. The BuildKit cache mount holds downloaded tarballs across
# builds; node_modules itself is committed to the layer. .npmrc pins
# ignore-scripts=true and the min-release-age cooldown.
COPY package.json package-lock.json .npmrc .nvmrc ./
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci --ignore-scripts --no-audit --no-fund

# Application sources (change frequently) come after the cached install layer.
# CHANGELOG.md is required: src/components/sidebar/VersionInfo.jsx imports it
# via `?raw`, so the build fails without it.
COPY CHANGELOG.md ./
COPY index.html vite.config.js tsconfig.json ./
COPY src/ ./src/
COPY public/ ./public/

RUN npm run build

###############################################################################
# Stage 2 - run the FastAPI app
###############################################################################
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Phlox" \
      org.opencontainers.image.description="Patient management with AI transcription and clinical notes - FastAPI backend plus prebuilt React frontend (Persian/RTL build)" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/AmiraliGhamkhar/phlox_persian"

# uv version must match [tool.uv].required-version in server/pyproject.toml.
COPY --from=ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c /uv /usr/local/bin/uv

WORKDIR /usr/src/app

# DOCKER_CONTAINER selects the Docker code path (no desktop passphrase prompt) and
# pins DATA_DIR/BUILD_DIR/TEMP_DIR to /usr/src/app (see server/constants.py).
# PATH picks up the uv-locked venv created below; PYTHONPATH makes `server`
# importable from any working directory; SERVER_HOST/PORT are the uvicorn bind
# address and port (override with -e at run time).
# UV_PYTHON_* stop uv from downloading a second CPython during the image build -
# this base image already ships the interpreter required by the lockfile.
# TIKTOKEN_CACHE_DIR points at a cache directory baked into the image (and owned
# by the runtime user) so tokenizer data never has to be fetched at run time.
# UV_LINK_MODE=copy is required with BuildKit cache mounts: the uv cache is not
# present at container run time, so installed files must be copied (not
# hard-linked) into the venv.
ENV DOCKER_CONTAINER=true \
    PATH=/usr/src/app/server/.venv/bin:$PATH \
    PYTHONPATH=/usr/src/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Must stay 0.0.0.0: docker-proxy connects to the container's bridge IP,
    # so a loopback bind inside the container would break published ports.
    # Exposure is controlled by the compose port mapping (loopback by
    # default) and the startup guard in server/server.py.
    SERVER_HOST=0.0.0.0 \
    PORT=5000 \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    TIKTOKEN_CACHE_DIR=/usr/src/app/.cache/tiktoken

# tesseract-ocr: image attachment OCR (server/nlp_tools/document_processing.py)
# tzdata: makes the TZ environment variable actually resolve to a local zone
# ca-certificates: TLS trust for outbound LLM / ASR / embedding requests
# The apt cache mounts keep package lists/debs warm across builds (and out of
# the final image); the explicit removal keeps the committed layer minimal.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tesseract-ocr \
        tesseract-ocr-eng \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create phlox user (uid:gid 1000:1000 - see the bind-mount note in docker-compose.yml)
RUN groupadd -g 1000 phlox \
    && useradd -m -u 1000 -g 1000 -s /usr/sbin/nologin phlox

# Application content
COPY --from=build /usr/src/app/build ./build
COPY CHANGELOG.md ./CHANGELOG.md
COPY server/pyproject.toml server/uv.lock server/.python-version ./server/

# Writable paths: data (DB + vectors + backups + logs), temp uploads, token cache.
RUN mkdir -p /usr/src/app/data /usr/src/app/temp /usr/src/app/.cache/tiktoken \
    && chown -R phlox:phlox /usr/src/app

# Install Python dependencies from the lockfile (rag + ocr extras via "docker",
# dev extras excluded) into /usr/src/app/server/.venv. The BuildKit cache mount
# keeps downloaded wheels warm across builds without bloating the image. The
# venv itself is committed to the layer so the runtime image needs no cache.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --directory server --locked --no-dev --extra docker

# Pre-cache the tiktoken encoding used by RAG chunking and letter budgeting.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chown -R phlox:phlox /usr/src/app/.cache

# Copy remaining server code (the venv above is not part of the build context).
COPY server/ ./server/

RUN chown -R phlox:phlox /usr/src/app

# Container health probe (no curl needed - stdlib only). 401/403 count as
# healthy: the app is up, an auth proxy is just doing its job.
RUN printf '%s\n' \
'"""Liveness probe for the Phlox container."""' \
'import os' \
'import sys' \
'import urllib.error' \
'import urllib.request' \
'' \
'port = os.environ.get("PORT", "5000")' \
'url = "http://127.0.0.1:" + port + "/api/dashboard/health"' \
'' \
'try:' \
'    with urllib.request.urlopen(url, timeout=5) as response:' \
'        sys.exit(0 if response.status == 200 else 1)' \
'except urllib.error.HTTPError as exc:' \
'    sys.exit(0 if exc.code in (401, 403) else 1)' \
'except Exception:' \
'    sys.exit(1)' \
> /usr/local/bin/phlox-healthcheck.py && chmod 0755 /usr/local/bin/phlox-healthcheck.py

USER phlox

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "/usr/local/bin/phlox-healthcheck.py"]

# uvicorn installs SIGTERM/SIGINT handlers and shuts down gracefully
# (timeout_graceful_shutdown=10), so forward the signal untouched.
STOPSIGNAL SIGTERM

# `python -m` keeps SERVER_HOST/PORT environment overrides working,
# unlike `uvicorn server.server:app` with an explicit --host/--port pair.
CMD ["python", "-m", "server.server"]
