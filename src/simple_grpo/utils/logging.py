import os
import json
from typing import Any


def write_generation_log(log_data: dict[str, Any], log_file: str) -> None:
    """
    Write generation log data to a text file.

    Args:
        log_data: dictionary containing prompt and generation data
        log_file: Path to output log file
    """
    with open(log_file, "a") as f:  # Append mode
        f.write(json.dumps(log_data, indent=2) + "\n---\n")  # Add separator


# Constants for PDF generation
def setup_training_log_directory(output_dir: str) -> str:
    """Creates and returns the path for the training log directory."""
    training_log_dir = os.path.join(output_dir, "training_logs")
    os.makedirs(training_log_dir, exist_ok=True)
    return training_log_dir


def setup_eval_directories(base_output_dir: str) -> tuple[str, str, str]:
    """Creates and returns paths for evaluation log directories (PDF and JSON)."""
    logs_dir = os.path.join(base_output_dir, "eval_logs")
    pdf_dir = os.path.join(logs_dir, "pdfs")
    json_dir = os.path.join(logs_dir, "json_reports")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    return logs_dir, pdf_dir, json_dir


def calculate_and_log_final_metrics(
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
