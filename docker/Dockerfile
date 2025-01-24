FROM ubuntu:24.04

WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN /root/Miniconda3-latest-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN echo 'conda activate main' >> ~/.bashrc
RUN /root/miniconda3/bin/conda create --name main python==3.12
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


COPY requirements.txt /requirements.txt
RUN /root/miniconda3/bin/conda run -n main pip install --no-cache-dir -r /requirements.txt

ENV WANDB_API_KEY=2aa2cc64e8422604fbbc4711274a7680f8ce951b

WORKDIR /workspace
