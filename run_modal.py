import modal
import subprocess

python_version = "3.11"
flash_attn_version = "2.6.3"
pytorch_version = "2.7.0"
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
flash_attn_release = "flash_attn-2.6.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.0")
    .run_commands(  # add flash-attn
        f"pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.9/{flash_attn_release}"
    )
    .pip_install(
        "transformers",
        "datasets",
        "torchvision",
        "qwen-vl-utils",
        "reportlab",
        "matplotlib",
        "numpy",
        "accelerate",
        "packaging",
        "ninja",
    )
    .pip_install("hf_xet")
    .run_commands(
        "git clone -b captcha --single-branch https://github.com/andersonbcdefg/simple-grpo.git ",
        "cd simple-grpo && pip install -e .",
        force_build=True,
    )
    .entrypoint([])
)
# image = modal.Image.debian_slim(python_version="3.11")

app = modal.App("deepseek-extended2")


@app.function(image=image, gpu="H100", timeout=60 * 60 * 5)
def run():
    # print(os.listdir("."))
    # print(os.listdir("/"))
    subprocess.run(["python", "main.py"], cwd="/simple-grpo")
