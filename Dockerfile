# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime AS builder
WORKDIR /
ENV FORCE_CUDA=1 MAX_JOBS=3 TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6" python_abi=cp310-cp310m
RUN apt-get update && apt-get install -y git

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
WORKDIR /
RUN --mount=type=bind,target=whls,from=builder apt-get update && apt-get install -y git wget && \
    git clone https://github.com/ShivamShrirao/diffusers && \
    cd diffusers/examples/dreambooth/ && \
    git checkout fbdf0a17055ffa34679cb34d986fabc1296d0785 && \
    pip install --no-cache-dir /diffusers triton==2.0.0 scipy pytorch-lightning bitsandbytes-cuda116 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -I markupsafe==2.0.1 OmegaConf && \
    cp train_dreambooth.py / && \
    rm -rf /var/lib/apt/lists/*
COPY start_training /start_training
WORKDIR /train
ENV HF_HOME=/train/.hub
