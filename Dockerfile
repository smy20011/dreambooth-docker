FROM nvcr.io/nvidia/pytorch:22.08-py3
WORKDIR /
ENV FORCE_CUDA=1 MAX_JOBS=3
RUN pip3 install -v git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers transformers ftfy scipy
RUN git clone https://github.com/ShivamShrirao/diffusers && \
    cd diffusers/examples/dreambooth/ && \
    pip install --no-cache-dir /diffusers triton==2.0.0.dev20220701 bitsandbytes && \
    pip install --no-cache-dir -r requirements.txt && \
    cp train_dreambooth.py /
WORKDIR /train
ENV HF_HOME=/train/.hub
