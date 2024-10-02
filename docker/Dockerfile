FROM python:3.13.0rc1-alpine3.20
#WORKDIR /usr/src/app
WORKDIR /workspace

COPY requirements.txt ./
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
