FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


COPY service/pyproject.toml service/uv.lock ./


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# 5. Копируем весь оставшийся код
COPY . .

CMD ["uv", "run", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
