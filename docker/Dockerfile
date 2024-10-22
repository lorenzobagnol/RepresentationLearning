FROM python:3.11.10-slim
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN chmod +x /run.sh
ENV WANDB_API_KEY=2aa2cc64e8422604fbbc4711274a7680f8ce951b
