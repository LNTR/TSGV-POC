FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/visual-preprocessing-service

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY TALL/implementation/visual-preprocessing-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY service_registry.py /app/service_registry.py
COPY TALL/implementation/visual-preprocessing-service/app ./app

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
