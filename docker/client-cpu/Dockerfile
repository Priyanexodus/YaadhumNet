# Dockerfile.client.cpu
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app
COPY .python-version pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra client --extra cpu --no-cache

COPY moon_fl/ ./moon_fl/
RUN uv sync --frozen --extra client --extra cpu --no-cache

# Bake CA certificate — SuperNode uses this to verify SuperLink TLS cert
COPY certs/ca.crt /app/certs/ca.crt

# Entrypoint — validates env vars and starts SuperNode
COPY docker/client/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"

# Required at runtime — no defaults
ENV SUPERLINK_URL=""
ENV DATA_PATH=""

ENTRYPOINT ["/app/entrypoint.sh"]
