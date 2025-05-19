import torch
from typing import Callable, Any, cast
from simple_grpo.typedefs import MessageListBatch
from simple_grpo.utils.plotter import plot_captcha_evaluation
from ..evaluator import RewardEvaluator, GUIEvaluator
from PIL import Image as PILImage
from simple_grpo.utils.reports import _add_completion_to_pdf


# Import constants from captcha_generator if needed for plotting logic
from simple_grpo.datasets.captcha_generator import (
    FINAL_DIM as CAPTCHA_FINAL_DIM,
    GRID_SIZE as CAPTCHA_GRID_SIZE,
    BANNER_ABS_HEIGHT as CAPTCHA_BANNER_ABS_HEIGHT,
    PADDING_SIZE as CAPTCHA_PADDING_SIZE,
    CELL_DIM as CAPTCHA_CELL_DIM,
)


def _process_single_completion_for_eval(
    completion_text: str,
    eval_class: RewardEvaluator,
    answer_data: Any,  # This is target_details for GUI, answer_str for clock/correlation
    device: str,
    story: list | None = None,  # Make it optional
    styles: dict | None = None,  # Make it optional
    completion_idx: int = 0,
    dataset_type: str = "gui",
    original_image_path: str | None = None,
    vis_image_path_for_pdf: str | None = None,
    gui_plotter: Callable | None = None,
    verbose: bool = False,  # Added verbose for plot_captcha_evaluation
) -> dict[str, float] | None:
    """
    Processes a single completion text for evaluation and optionally adds it to a PDF story.
    Returns a dictionary of metric scores for this single completion.
    """
    if (
        dataset_type == "gui" or dataset_type == "captcha"
    ):  # Captcha also uses dict answer_data
        current_answers_list = [answer_data]
        current_completions_list = [[{"content": completion_text}]]
    else:  # clock, correlation
        # For clock/correlation, answer_data is the answer string.
        current_answers_list = [answer_data]
        current_completions_list = [[{"content": completion_text}]]

    # Get rewards and metrics for this single completion
    # The evaluator's compute_rewards should return rewards_per_func and metrics
    # rewards_per_func will be a tensor for this one completion, e.g., shape [1, num_reward_components]
    # metrics will be a dict like {'metric_name': value_tensor}
    rewards_per_func_single, metrics_single_dict_tensors = eval_class.compute_rewards(
        prompts=None,  # Not always needed by evaluator if context is in answer_data
        completions=cast(MessageListBatch, current_completions_list),
        answers=current_answers_list,
        device=device,  # device might be used by evaluator internally
    )

    # Convert metric tensors to scalar floats and ensure all are included
    # Also, get the reward breakdown for PDF logging if story is present
    processed_metrics_for_return = {}
    if metrics_single_dict_tensors and isinstance(metrics_single_dict_tensors, dict):
        for k, v_tensor in metrics_single_dict_tensors.items():
            if torch.is_tensor(v_tensor) and v_tensor.numel() == 1:
                processed_metrics_for_return[k] = v_tensor.item()
            elif isinstance(v_tensor, (float, int)):
                processed_metrics_for_return[k] = v_tensor
            # else: might be other types of metrics not directly plottable/averageable

    # Get reward breakdown if needed (e.g., for PDF)
    # rewards_per_func_single should be for one sample, e.g., shape [1, num_reward_components]
    # We need to pass the actual reward scores for this one completion to get_reward_breakdown
    reward_scores_for_breakdown = rewards_per_func_single[
        0
    ]  # Get the tensor for the first (only) sample

    # Add overall reward to the metrics
    total_reward_single = reward_scores_for_breakdown.sum().item()
    processed_metrics_for_return["reward"] = (
        total_reward_single  # Ensure 'reward' key exists
    )

    # --- PDF Logging Section ---
    # Only attempt PDF operations if story and styles are provided
    if story is not None and styles is not None:
        reward_breakdown_for_pdf = eval_class.get_reward_breakdown(
            reward_scores_for_breakdown
        )

        # Add total_reward_single to the dictionary that will be passed as metrics to _add_completion_to_pdf
        reward_breakdown_for_pdf["total_reward_this_completion"] = total_reward_single

        img_path_for_pdf_entry = None
        captcha_stats = None

        if (
            dataset_type == "gui"
            and original_image_path
            and vis_image_path_for_pdf
            and gui_plotter
        ):
            try:
                # Plot click for GUI task if a plotter is provided
                if isinstance(
                    eval_class, GUIEvaluator
                ):  # Check if it's the right evaluator
                    parsed_click = eval_class._extract_coordinates(
                        completion_text
                    )  # Protected access, but used in main.py
                    if parsed_click:
                        with PILImage.open(original_image_path) as pil_img:
                            plot_data = [
                                {
                                    "name": "VLM Click",
                                    "center_x": parsed_click[0],
                                    "center_y": parsed_click[1],
                                    "is_truth": False,
                                }
                            ]
                            # GUIGenerator.plot_predictions returns a PIL Image
                            img_w_click = gui_plotter(pil_img, plot_data, pred_color="red")
                            img_w_click.save(vis_image_path_for_pdf)
                            img_w_click.close()
                        img_path_for_pdf_entry = vis_image_path_for_pdf
                    else:
                        # If click not parsed, use original image for PDF (or None if vis_image_path_for_pdf was for specific click)
                        img_path_for_pdf_entry = (
                            original_image_path  # Or handle as per main.py logic
                        )
                else:
                    img_path_for_pdf_entry = original_image_path  # Fallback for non-GUIEvaluator or if no click
            except Exception as plot_err:
                if verbose:  # Assuming verbose is accessible or passed
                    print(
                        f"  Warning: Error plotting click for PDF (utils): {plot_err}"
                    )
                img_path_for_pdf_entry = original_image_path  # Fallback
        elif (
            dataset_type == "captcha" and original_image_path and vis_image_path_for_pdf
        ):
            # CAPTCHA specific PDF logging for evaluation
            img_path_for_pdf_entry = None

            # Create a temporary CaptchaEvaluator to extract clicks
            from simple_grpo.evaluator import CaptchaEvaluator

            temp_captcha_evaluator = CaptchaEvaluator()
            predicted_clicks = temp_captcha_evaluator._extract_click_calls(
                completion_text
            )

            # Calculate CAPTCHA accuracy stats for the plain English summary
            if "target_squares_boolean" in answer_data:
                target_squares_boolean = answer_data["target_squares_boolean"]
                total_targets = sum(1 for x in target_squares_boolean if x)
                total_clicks = len(predicted_clicks)

                # Manually calculate TP, FP, FN for clarity
                tp = 0  # True positives
                fp = 0  # False positives
                clicked_squares = set()

                # For each click, determine which square it falls in and if it's a target
                for px, py in predicted_clicks:
                    # Convert pixel coordinates to grid cell
                    # Calculate grid parameters using constants from CAPTCHA generator
                    content_width_unpadded = CAPTCHA_CELL_DIM * CAPTCHA_GRID_SIZE
                    content_height_unpadded = (
                        CAPTCHA_BANNER_ABS_HEIGHT + content_width_unpadded
                    )
                    pre_resize_width = content_width_unpadded + 2 * CAPTCHA_PADDING_SIZE
                    pre_resize_height = (
                        content_height_unpadded + 2 * CAPTCHA_PADDING_SIZE
                    )
                    scale_x = CAPTCHA_FINAL_DIM / pre_resize_width
                    scale_y = CAPTCHA_FINAL_DIM / pre_resize_height
                    final_grid_start_x = CAPTCHA_PADDING_SIZE * scale_x
                    final_grid_start_y = (
                        CAPTCHA_PADDING_SIZE + CAPTCHA_BANNER_ABS_HEIGHT
                    ) * scale_y
                    final_cell_width = CAPTCHA_CELL_DIM * scale_x
                    final_cell_height = CAPTCHA_CELL_DIM * scale_y

                    # Determine grid position
                    if (
                        px < final_grid_start_x
                        or py < final_grid_start_y
                        or px
                        >= final_grid_start_x + final_cell_width * CAPTCHA_GRID_SIZE
                        or py
                        >= final_grid_start_y + final_cell_height * CAPTCHA_GRID_SIZE
                    ):
                        # Click is outside the grid
                        fp += 1
                        continue

                    # Calculate which cell was clicked
                    col = int((px - final_grid_start_x) / final_cell_width)
                    row = int((py - final_grid_start_y) / final_cell_height)
                    col = max(0, min(col, CAPTCHA_GRID_SIZE - 1))
                    row = max(0, min(row, CAPTCHA_GRID_SIZE - 1))

                    square_idx = row * CAPTCHA_GRID_SIZE + col

                    # Only count unique square clicks (first click per square)
                    if square_idx not in clicked_squares:
                        clicked_squares.add(square_idx)
                        # Check if it's a target square
                        if (
                            square_idx < len(target_squares_boolean)
                            and target_squares_boolean[square_idx]
                        ):
                            tp += 1
                        else:
                            fp += 1

                # Calculate false negatives (missed targets)
                fn = total_targets - tp

                # Store the stats for the plain English explanation
                captcha_stats = {
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "total_targets": total_targets,
                    "total_clicks": total_clicks,
                }

            # Plot the visualization with the clicks
            plot_captcha_evaluation(
                base_image_path=original_image_path,
                predicted_clicks=predicted_clicks,
                target_squares_boolean=answer_data["target_squares_boolean"],
                output_path=vis_image_path_for_pdf,
                verbose=verbose,
            )
            img_path_for_pdf_entry = vis_image_path_for_pdf

        elif dataset_type != "gui" and original_image_path:
            img_path_for_pdf_entry = original_image_path

        # Add the completion to the PDF with the appropriate image and CAPTCHA stats
        _add_completion_to_pdf(
            story,
            styles,
            completion_text,
            metrics=reward_breakdown_for_pdf,  # Pass the breakdown as metrics
            completion_idx=completion_idx,
            dataset_type=dataset_type,
            image_path_for_completion_pdf=img_path_for_pdf_entry,  # Pass the plotted image
            captcha_data=captcha_stats,  # Pass the captcha statistics
        )

    # --- End PDF Logging Section ---

    return processed_metrics_for_return
