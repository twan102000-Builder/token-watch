FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tokenwatch ./tokenwatch

RUN pip install --no-cache-dir .

EXPOSE 9000

CMD ["python", "-m", "tokenwatch.hub", "--host", "0.0.0.0", "--port", "9000"]
