FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/training-service

COPY TGN/implementation/training-service/requirements.txt ./requirements.txt
RUN grep -v '^torch==' requirements.txt > requirements.docker.txt
RUN pip install --no-cache-dir -r requirements.docker.txt
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    --retries 10 \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1

COPY service_registry.py /app/service_registry.py
COPY TGN/implementation/training-service/app ./app

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
