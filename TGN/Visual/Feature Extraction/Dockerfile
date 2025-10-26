FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
RUN pip install --no-cache-dir -r requirements.txt

VOLUME [ "/data" ]
