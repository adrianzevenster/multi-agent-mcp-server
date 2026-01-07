FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.api.txt /app/requirements.api.txt
RUN pip install --no-cache-dir -r /app/requirements.api.txt

COPY . /app

CMD ["sh", "-c", "uvicorn app.mcp.mcp_http_api:app --host 0.0.0.0 --port ${PORT}"]
