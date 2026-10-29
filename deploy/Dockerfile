FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tokenwatch ./tokenwatch

RUN pip install --no-cache-dir .

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/health')" || exit 1

CMD ["python", "-m", "tokenwatch.hub", "--host", "0.0.0.0", "--port", "9000"]
