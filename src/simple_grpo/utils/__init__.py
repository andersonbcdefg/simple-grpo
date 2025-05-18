import re
import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any
from qwen_vl_utils import process_vision_info
from transformers.tokenization_utils_base import BatchEncoding
from simple_grpo.datasets.gui_generator import (
    GUIGenerator as GUIGenerator,
)  # For plot_predictions type hint if GUI specific logic

MAX_COMPLETIONS_PER_PAGE_PDF = 2
MAX_PROMPT_LENGTH_PDF = 300  # Add the missing constant definition
MAX_ANSWER_LENGTH_PDF = 200
MAX_COMPLETION_LENGTH_PDF = 500

####################
## MISC FUNCTIONS ##
####################


def set_dtype(encoding: BatchEncoding, dtype: torch.dtype | str):
    for k, v in encoding.items():
        if isinstance(v, torch.Tensor):
            encoding[k] = v.to(dtype)
    return encoding


def clean_spaces_preserve_newlines(text):
    # Replace multiple spaces with a single space, but preserve newlines
    lines = text.split("\n")  # Split by newlines
    cleaned_lines = [
        " ".join(re.split(r"\s+", line)).strip() for line in lines
    ]  # Remove extra spaces in each line
    return "\n".join(cleaned_lines)  # Join the lines back with newlines


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple libraries.

    This function sets consistent random seeds for Python's random module,
    NumPy, PyTorch (both CPU and CUDA), and configures CUDNN for deterministic
    operation. This ensures reproducible results across multiple runs.

    Args:
        seed: The random seed to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Add set_seed (alias for seed_everything)
set_seed = seed_everything


def write_generation_log(log_data: dict[str, Any], log_file: str) -> None:
    """
    Write generation log data to a text file.

    Args:
        log_data: dictionary containing prompt and generation data
        log_file: Path to output log file
    """
    with open(log_file, "a") as f:  # Append mode
        f.write(json.dumps(log_data, indent=2) + "\n---\n")  # Add separator


####################################################################################
## Copied Directly from TRL -> generate log probs per token                 ########
## https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py ########
####################################################################################


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = (
            selected_logits - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(
            logits, index
        ):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
    ).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens


def get_per_token_logps_vl(
    model, input_ids, attention_mask, image_path, tokenizer, logits_to_keep, prompt
):
    """
    We have the input ids - all the correct tokens including all chate templates/special tokens etc
    We just need to include the image - and have the same sort of obj to pass to the model to generate
    the logits

    So lets generate with the image


    resulting to a very non-generic way to do this - TODO: make this better
    """

    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. You are an expert image analyst.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False, padding_side="left"
    )
    image_inputs, video_inputs = process_vision_info(conversation)  # type: ignore -

    prompt_inputs = (
        tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )
        .to(model.device)
        .to(model.dtype)
    )

    # Repeat input tensors for batch generation
    batched_prompt_inputs = {}
    for key, value in prompt_inputs.items():
        if torch.is_tensor(value):
            batched_prompt_inputs[key] = value.repeat(
                input_ids.shape[0], *([1] * (value.dim() - 1))
            )
        else:
            # Handle non-tensor items if necessary, otherwise just copy
            batched_prompt_inputs[key] = value

    batched_prompt_inputs["input_ids"] = input_ids
    batched_prompt_inputs["attention_mask"] = attention_mask

    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    # logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = model(**batched_prompt_inputs).logits
    logits = logits[
        :, :-1, :
    ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(
        logits, input_ids
    )  #  compute logprobs for the input tokens


########################
## PDF/HTML/TEXT STUFF ##
########################


# Constants for PDF generation
def _setup_training_log_directory(output_dir: str) -> str:
    """Creates and returns the path for the training log directory."""
    training_log_dir = os.path.join(output_dir, "training_logs")
    os.makedirs(training_log_dir, exist_ok=True)
    return training_log_dir


def _setup_eval_directories(base_output_dir: str) -> tuple[str, str, str]:
    """Creates and returns paths for evaluation log directories (PDF and JSON)."""
    logs_dir = os.path.join(base_output_dir, "eval_logs")
    pdf_dir = os.path.join(logs_dir, "pdfs")
    json_dir = os.path.join(logs_dir, "json_reports")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    return logs_dir, pdf_dir, json_dir


def _calculate_and_log_final_metrics(
    all_avg_scores: dict, json_dir: str, round_num: int, verbose: bool
):
    """Saves combined average scores (overall, normal, hard) to JSON and prints if verbose."""
    # Save average scores to a JSON file
    avg_scores_path = os.path.join(json_dir, f"average_scores_round_{round_num}.json")
    with open(avg_scores_path, "w") as f:
        # Log all average scores (overall, normal, hard)
        json.dump(all_avg_scores, f, indent=4)

    if verbose:
        print(f"\n--- Evaluation Results (Round {round_num}) ---")
        print(f"Average scores saved to {avg_scores_path}")
        print("Average Scores Breakdown:")
        # Nicely print the different groups if they exist
        print("  --- Overall --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_overall_"):
                print(f"    {name.replace('avg_overall_', ''):<35}: {value:.4f}")
        print("  --- Normal Subset --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_normal_"):
                print(f"    {name.replace('avg_normal_', ''):<35}: {value:.4f}")
        if not any(k.startswith("avg_normal_") for k in all_avg_scores):
            print("    (No normal examples in this evaluation)")
        print("  --- Hard Subset --- ")
        for name, value in all_avg_scores.items():
            if name.startswith("avg_hard_"):
                print(f"    {name.replace('avg_hard_', ''):<35}: {value:.4f}")
        if not any(k.startswith("avg_hard_") for k in all_avg_scores):
            print("    (No hard examples in this evaluation)")
        print("-" * 40)
