FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app/training-service

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY TALL/implementation/training-service/requirements.txt ./requirements.txt
RUN grep -v '^torch==' requirements.txt > requirements.docker.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir -r requirements.docker.txt
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1

COPY service_registry.py /app/service_registry.py
COPY TALL/implementation/tall_model.py /app/tall_model.py
COPY TALL/implementation/tall_runtime.py /app/tall_runtime.py
COPY TALL/implementation/tall_text.py /app/tall_text.py
COPY TALL/implementation/training-service/app ./app

EXPOSE 8004

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
