# =============================================================================
# ReactorGuard API — Multi-stage Dockerfile
# =============================================================================
# Stage 1 (builder): installs all Python dependencies in an isolated layer.
# Stage 2 (runtime): copies only the installed packages and app code — no
#   build tools, no pip cache, no unnecessary files — minimising the attack
#   surface for a safety-critical production container.
#
# Security hardening:
#   - Runs as non-root user (appuser, UID 1000) — limits blast radius if the
#     container is compromised.
#   - HEALTHCHECK lets Kubernetes liveness/readiness probes use the same
#     /health endpoint defined in the FastAPI app.
#   - python:3.11-slim base is kept minimal; Trivy scans in CI enforce
#     0 CRITICAL CVEs before any image is pushed to GCR.
# =============================================================================

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed to compile some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the manifest first — Docker layer cache means this layer
# is only invalidated when pyproject.toml changes, not when source code changes.
COPY pyproject.toml .

# Stub package directories so pip can resolve the editable install metadata
RUN mkdir -p api ml data

# Install runtime dependencies only (no [dev] extras) into a prefix directory
# so we can copy the whole tree cleanly in stage 2.
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user and group
RUN groupadd --gid 1000 appgroup && \
    useradd  --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY api/ ./api/
COPY ml/  ./ml/
COPY data/ ./data/

# Transfer ownership so the non-root user can read the app files
RUN chown -R appuser:appgroup /app

# Drop privileges — all subsequent commands run as appuser
USER appuser

# Expose the port Uvicorn listens on
EXPOSE 8000

# HEALTHCHECK: Kubernetes uses this to decide whether to route traffic to the
# pod. A 200 response from /health means the model is loaded and the API is
# ready. Interval/timeout are tuned for a safety-critical environment where
# stale pods must be detected quickly.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start Uvicorn with a single worker (horizontal scaling handled by K8s HPA)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
