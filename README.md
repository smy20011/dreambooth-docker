# DreamBooth local docker file for windows/linux

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.

The training script in this repo is adapted from ShivamShrirao's diffuser repo. See [here](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) for detailed training command.

Docker file copy the ShivamShrirao's train_dreambooth.py to root directory. Replace any train_dreambooth.py in original doc with /train_dreambooth.py. Eg, if you want to run `accelerate launch train_dreambooth.py` you need to run following

```bash
sudo docker run -it --gpus=all --ipc=host -v $(pwd):/train -e HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)  smy20011/dreambooth:latest \
  accelerate launch /train_dreambooth.py (your arguments here)
```

## Running locally 

You need to install WSL with docker support. For linux users, make sure you have NVIDIA driver installed and have latest docker installed. If you still cannot access GPU, make sure you have [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

For Windows users, follow the guide [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [here](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers). You don't need to do the "CUDA Support for WSL2 section".
You can also follow this [Youtube](https://www.youtube.com/watch?v=YozfiLI1ogY) video for reference as well.

Open the WSL terminal in Windows or open the terminal in Linux. Makesure you have [huggingface-cli](https://huggingface.co/docs/huggingface_hub/quick-start) installed.

### Dog toy example

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree. 

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

<br>

Now let's get our dataset. Download images from [here](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ) and save them in a directory. This will be our training data.

`cd` to the directory in your terminal. Run following command


```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

sudo docker run -it --gpus=all --ipc=host -v $(pwd):/train -e HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)  smy20011/dreambooth:latest \
  accelerate launch /train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.
According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

sudo docker run -it --gpus=all --ipc=host -v $(pwd):/train -e HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)  smy20011/dreambooth:latest \
  accelerate launch /train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Training on a 16GB GPU:

With the help of gradient checkpointing and the 8-bit optimizer from bitsandbytes it's possible to run train dreambooth on a 16GB GPU.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

sudo docker run -it --gpus=all --ipc=host -v $(pwd):/train -e HUGGING_FACE_HUB_TOKEN=$(cat ~/.huggingface/token)  smy20011/dreambooth:latest \
  accelerate launch /train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```
