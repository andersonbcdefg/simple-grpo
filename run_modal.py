import os
import subprocess
import modal

nums = lambda x: "".join([ch for ch in x if ch.isdigit()])

python_version = "3.11"
flash_attn_version = "2.6.3"
pytorch_version = "2.7.0"
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
flash_attn_release = f"flash_attn-2.6.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.0")
    .run_commands(  # add flash-attn
        f"pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.9/{flash_attn_release}"
    ).pip_install(
        "transformers",
        "datasets",
        "torchvision",
        "qwen-vl-utils",
        "reportlab",
        "matplotlib",
        "numpy",
        "accelerate",
        "packaging",
        "ninja"
    ).run_commands(
        "git clone -b captcha --single-branch https://github.com/andersonbcdefg/simple-grpo.git ",
        force_build=True
    ).workdir("simple-grpo")
)


app = modal.App('deepseek-extended')

@app.function(image=image, gpu="H100")
def hello():
    subprocess.run(["python", "main.py"])
