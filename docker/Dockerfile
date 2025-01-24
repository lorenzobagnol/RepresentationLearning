FROM ubuntu:24.04

RUN apt update && apt install -y python3 python3-pip
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt
ENV WANDB_API_KEY=2aa2cc64e8422604fbbc4711274a7680f8ce951b

WORKDIR /workspace
