FROM python:3.13.0rc1-alpine3.20
#WORKDIR /usr/src/app
WORKDIR /workspace
ENV WANDB_API_KEY=2aa2cc64e8422604fbbc4711274a7680f8ce951b
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
