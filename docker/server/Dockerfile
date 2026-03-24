FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY .python-version pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra server --no-cache

COPY moon_fl/ ./moon_fl/
RUN uv sync --frozen --extra server --no-cache

COPY docker/server/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9091 9092 9093

ENTRYPOINT ["/app/entrypoint.sh"]
