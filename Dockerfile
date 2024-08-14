FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python==4.10.0.84 imageio==2.35.0 imageio-ffmpeg==0.5.1 ffmpeg-python==0.2.0 av==12.3.0 runpod==1.7.0 \
    xformers==0.0.25 open_clip_torch==2.26.1 torchsde==0.2.6 einops==0.8.0 diffusers==0.30.0 transformers==4.44.0 accelerate==0.33.0 && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone -b tost https://github.com/camenduru/ComfyUI-LLaVA-OneVision /content/ComfyUI/custom_nodes/ComfyUI-LLaVA-OneVision && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.sft -d /content/ComfyUI/models/unet -o flux1-dev.sft && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d /content/ComfyUI/models/clip -o clip_l.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp16.safetensors -d /content/ComfyUI/models/clip -o t5xxl_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /content/ComfyUI/models/vae -o ae.sft && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/aesthetic10k.safetensors -d /content/ComfyUI/models/loras -o advokat_aesthetic10k.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_anime_lora.safetensors -d /content/ComfyUI/models/loras -o xlabs_anime.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_art_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_art.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_disney_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_disney.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_mjv6_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_mjv6.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_realism_lora_comfui.safetensors -d /content/ComfyUI/models/loras -o xlabs_realism.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/xlabs_flux_scenery_lora_comfyui.safetensors -d /content/ComfyUI/models/loras -o xlabs_scenery.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux_dev_frostinglane_araminta_k.safetensors -d /content/ComfyUI/models/loras -o alvdansen_frostinglane_araminta_k.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/resolve/main/model.safetensors -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/tokenizer.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/vocab.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/trainer_state.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o trainer_state.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/resolve/main/training_args.bin -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o training_args.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/tokenizer_config.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/merges.txt -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/added_tokens.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o added_tokens.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/special_tokens_map.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/config.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/4bit/llava-onevision-qwen2-0.5b-si/raw/main/generation_config.json -d /content/ComfyUI/models/LLM/LLaVA-OneVision/llava-onevision-qwen2-0.5b-si -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/config.json -d /content/ComfyUI/models/LLM/siglip-so400m-patch14-384 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/model.safetensors -d /content/ComfyUI/models/LLM/siglip-so400m-patch14-384 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/preprocessor_config.json -d /content/ComfyUI/models/LLM/siglip-so400m-patch14-384 -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/special_tokens_map.json -d /content/ComfyUI/models/LLM/siglip-so400m-patch14-384 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/resolve/main/spiece.model -d /content/ComfyUI/models/LLM/siglip-so400m-patch14-384 -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/google/siglip-so400m-patch14-384/raw/main/tokenizer_config.json -d /content/ComfyUI/models/LLM/siglip-so400m-patch14-384 -o tokenizer_config.json

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py