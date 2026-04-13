FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/inference-service

COPY inference-service/requirements.txt ./requirements.txt
RUN grep -v '^torch==' requirements.txt > requirements.docker.txt
RUN pip install --no-cache-dir -r requirements.docker.txt
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1

COPY service_registry.py /app/service_registry.py
COPY tall_model.py /app/tall_model.py
COPY tall_runtime.py /app/tall_runtime.py
COPY tall_text.py /app/tall_text.py
COPY inference-service/app ./app

EXPOSE 8005

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
