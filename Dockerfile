FROM ubuntu:20.04
RUN apt update && apt install htop git wget vim curl -y

WORKDIR /root/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
RUN /root/Miniconda3-py38_4.10.3-Linux-x86_64.sh -b && eval "$(/root/miniconda3/bin/conda shell.bash hook)" && /root/miniconda3/bin/conda clean -afy
RUN /root/miniconda3/bin/conda init
RUN echo 'conda activate main' >> ~/.bashrc
RUN /root/miniconda3/bin/conda create --name main python==3.9
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


RUN /root/miniconda3/bin/conda run -n main pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
RUN /root/miniconda3/bin/conda run -n main pip install numpy==1.26.4
RUN /root/miniconda3/bin/conda run -n main pip install matplotlib==3.8.0
RUN /root/miniconda3/bin/conda run -n main pip install gym==0.23.1
RUN apt-get install g++ -y
RUN /root/miniconda3/bin/conda run -n main pip install nes-py
RUN /root/miniconda3/bin/conda run -n main pip install gym-super-mario-bros
RUN /root/miniconda3/bin/conda run -n main pip install imageio==2.34.1
RUN /root/miniconda3/bin/conda run -n main pip install pillow==10.3.0
RUN /root/miniconda3/bin/conda run -n main pip install albumentations==1.0.3
RUN /root/miniconda3/bin/conda run -n main pip install tqdm==4.66.4
RUN /root/miniconda3/bin/conda run -n main pip install pandas==2.2.1
RUN /root/miniconda3/bin/conda run -n main pip install seaborn
RUN /root/miniconda3/bin/conda run -n main pip install scikit-learn
RUN /root/miniconda3/bin/conda run -n main pip install SciPy==1.12.0
RUN /root/miniconda3/bin/conda run -n main apt-get install unzip
RUN /root/miniconda3/bin/conda run -n main pip install gdown
RUN /root/miniconda3/bin/conda run -n main pip install umap-learn==0.3.10
RUN /root/miniconda3/bin/conda run -n main pip install opencv-python==4.8.1.78

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN /root/miniconda3/bin/conda run -n main pip install "gym[atari, accept-rom-license]"

WORKDIR /workspace