FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

COPY .python-version pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra client --no-cache

COPY moon_fl/ ./moon_fl/
RUN uv sync --frozen --extra client --no-cache

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
