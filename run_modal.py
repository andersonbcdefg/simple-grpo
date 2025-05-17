import os
import subprocess
import modal

image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "git", "curl"
).run_commands(
    "git clone -b captcha --single-branch https://github.com/andersonbcdefg/simple-grpo.git ",
).pip_install(
    "torch==2.7.0",
    "transformers"
).workdir("simple-grpo")

app = modal.App('deepseek-extended')

@app.function(image=image, gpu="H100")
def hello():
    subprocess.run(["python", "main.py"])
