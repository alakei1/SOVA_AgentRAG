FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Копируем файлы зависимостей (pyproject.toml, uv.lock) из корня проекта
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости и сам пакет (editable)
RUN uv sync --frozen --no-dev

# Копируем остальной код (включая папку service/)
COPY . .

# Запускаем приложение
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]