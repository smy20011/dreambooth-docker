# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel AS builder
WORKDIR /
ENV FORCE_CUDA=1 MAX_JOBS=3 TORCH_CUDA_ARCH_LIST="5.0;5.2;6.0;6.1+PTX;7.0;7.5+PTX;8.0;8.6+PTX" python_abi=cp37-cp37m
RUN apt-get update && apt-get install -y git
RUN pip3 wheel -v git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers
RUN pip3 wheel bitsandbytes 

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
WORKDIR /
ADD https://api.github.com/repos/ShivamShrirao/diffusers/git/refs/heads/main /version.json
RUN --mount=type=bind,target=whls,from=builder apt-get update && apt-get install -y git && \
    git clone https://github.com/ShivamShrirao/diffusers && \
    cd diffusers/examples/dreambooth/ && \
    pip install --no-cache-dir /diffusers triton==2.0.0.dev20220701 /whls/xformers*.whl /whls/bitsandbytes*.whl && \
    pip install --no-cache-dir -r requirements.txt && \
    cp train_dreambooth.py / && \
    rm -rf /var/lib/apt/lists/*
COPY start_training /start_training
# Fix waifu diffusion training.
RUN pip install scipy
WORKDIR /train
ENV HF_HOME=/train/.hub
