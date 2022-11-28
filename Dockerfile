# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel AS builder
WORKDIR /
ENV FORCE_CUDA=1 MAX_JOBS=3 TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6" python_abi=cp37-cp37m
RUN apt-get update && apt-get install -y git
RUN pip3 wheel -v git+https://github.com/facebookresearch/xformers
RUN pip3 wheel bitsandbytes 

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
WORKDIR /
RUN --mount=type=bind,target=whls,from=builder apt-get update && apt-get install -y git wget && \
    git clone https://github.com/ShivamShrirao/diffusers && \
    cd diffusers/examples/dreambooth/ && \
    git checkout 4affee && \
    pip install --no-cache-dir /diffusers triton==2.0.0.dev20220701 /whls/xformers*.whl /whls/bitsandbytes*.whl scipy pytorch-lightning && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -I markupsafe==2.0.1 OmegaConf && \
    cp train_dreambooth.py / && \
    rm -rf /var/lib/apt/lists/*
COPY start_training /start_training
WORKDIR /train
ENV HF_HOME=/train/.hub
