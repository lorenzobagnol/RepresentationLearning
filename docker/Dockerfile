FROM ubuntu:24.04

RUN apt update && apt install python3 -y
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
ENV WANDB_API_KEY=2aa2cc64e8422604fbbc4711274a7680f8ce951b

WORKDIR /workspace
