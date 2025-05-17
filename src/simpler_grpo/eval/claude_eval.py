import os
import json
import argparse
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import anthropic

# Assuming these modules are in the same environment/PYTHONPATH
import rldatasets
import evaluator
import utils # We'll use some helpers from here

# --- Claude API Configuration ---
# User should ensure ANTHROPIC_API_KEY is set in their environment
# Model can be changed if a specific "3.7" version ID is available.
# Using claude-3-5-sonnet-20240620 as it's the latest strong multimodal model.
CLAUDE_MODEL_ID = "claude-3-7-sonnet-20250219" 

def parse_args():
    parser = argparse.ArgumentParser(description="Claude Sonnet 3.5 Evaluation Script")
    
    # Dataset and Output
    parser.add_argument("--dataset_type", type=str, default="gui", choices=["clock", "correlation", "gui"], help="Type of dataset to use")
    parser.add_argument("--output_dir", type=str, default="claude_output", help="Directory to save outputs, including claude_eval.json")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Generation parameters for Claude
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for Claude")
    parser.add_argument("--num_chains_eval", type=int, default=1, help="Number of parallel completions to request from Claude per example (be mindful of cost/rate limits)")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length for Claude (max_tokens)")
    
    # Seed (for dataset loading, though Claude generation itself isn't seeded this way)
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed for dataset shuffling etc.")

    args = parser.parse_args()
    return args

def get_image_media_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".gif":
        return "image/gif"
    elif ext == ".webp":
        return "image/webp"
    else:
        # Fallback, though Claude might be specific. PNG is generally safe.
        print(f"Warning: Unknown image type {ext}, defaulting to image/png. Consider converting to PNG/JPEG.")
        return "image/png"

def image_to_base64(image_path):
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA': # Convert RGBA to RGB if that's an issue for Claude
            img = img.convert('RGB')
        
        buffered = BytesIO()
        img_format = img.format if img.format else 'PNG' # Default to PNG if format is not in metadata
        if img_format.upper() == 'JPEG':
             media_type = "image/jpeg"
        elif img_format.upper() == 'PNG':
             media_type = "image/png"
        else: # If it's something else, save as PNG
            img_format = 'PNG'
            media_type = "image/png"

        img.save(buffered, format=img_format)
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        return img_base64, media_type
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

async def generate_claude_completions(
    client: anthropic.AsyncAnthropic, # Use Async client for potential parallel chains
    image_path: str,
    prompt_text: str,
    args: argparse.Namespace
) -> list[str]:
    """
    Generate completions using Claude API.
    """
    completions_text = []
    
    img_base64, media_type = image_to_base64(image_path)
    if not img_base64:
        return [f"Error: Could not process image {image_path}"] * args.num_chains_eval

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_base64,
                    },
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # System prompt can be added if desired, similar to Qwen in main.py
    system_prompt = "You are an expert image analyst. Please follow the instructions carefully."

    try:
        # For num_chains_eval > 1, you might make parallel API calls.
        # For simplicity here, let's do them sequentially or assume num_chains_eval=1 for now.
        # If using async client, can use asyncio.gather for multiple chains.
        # For now, let's assume a loop for multiple chains for simplicity without full async.
        # For true parallelism, asyncio.gather with multiple client.messages.create calls is better.
        # This example will run them sequentially if num_chains_eval > 1.

        for _ in range(args.num_chains_eval):
            response = await client.messages.create(
                model=CLAUDE_MODEL_ID,
                max_tokens=args.max_completion_length,
                temperature=args.temperature,
                system=system_prompt,
                messages=messages,
            )
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                # Assuming the first content block of type 'text' is the response
                text_response = ""
                for block in response.content:
                    if block.type == "text":
                        text_response += block.text
                completions_text.append(text_response.strip())
            else:
                completions_text.append("Error: Empty or invalid response from Claude.")
                
    except anthropic.APIError as e:
        print(f"Anthropic API Error: {e}")
        completions_text.extend([f"Error: API call failed - {e}"] * (args.num_chains_eval - len(completions_text)))
    except Exception as e:
        print(f"Error during Claude completion: {e}")
        completions_text.extend([f"Error: SDK/Request failed - {e}"] * (args.num_chains_eval - len(completions_text)))
        
    # Ensure we return the expected number of completions, even if they are error messages
    while len(completions_text) < args.num_chains_eval:
        completions_text.append("Error: Generation failed to produce output.")
        
    return completions_text

async def run_claude_evaluation(args: argparse.Namespace):
    """
    Main evaluation loop using Claude.
    """
    utils.seed_everything(args.seed)
    
    try:
        client = anthropic.AsyncAnthropic() # API key from ANTHROPIC_API_KEY env var
    except Exception as e:
        print(f"Failed to initialize Anthropic client: {e}. Make sure ANTHROPIC_API_KEY is set.")
        return

    print(f"Loading dataset: {args.dataset_type}")
    # Get only the test_loader
    _, test_loader = rldatasets.get_dataloaders(args.dataset_type, eval_batch_size=1) # Batch size 1 for eval usually

    print(f"Loading evaluator for: {args.dataset_type}")
    eval_class = evaluator.get_evaluator(args.dataset_type)

    # Setup output directories (minimal version)
    os.makedirs(args.output_dir, exist_ok=True)
    # Mimic structure if utils expect it for saving, or save directly
    eval_json_dir = os.path.join(args.output_dir, "json_reports") 
    os.makedirs(eval_json_dir, exist_ok=True)


    print("Running evaluation on test set using Claude Sonnet 3.5...")
    
    num_examples = 0
    num_chains_processed_overall = 0
    num_chains_processed_normal = 0
    num_chains_processed_hard = 0
    aggregated_metrics_sum_overall = defaultdict(float)
    aggregated_metrics_sum_normal = defaultdict(float)
    aggregated_metrics_sum_hard = defaultdict(float)
    
    test_loader.reset()

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating with Claude")):
        is_hard_example = False
        if args.dataset_type == 'gui':
            img_path, target_details_dict = batch
            prompt_to_use = target_details_dict['dynamic_prompt']
            answer_for_eval = target_details_dict
            is_hard_example = target_details_dict.get('is_hard', False)
        else: # clock, correlation
            img_path, answer_string = batch
            prompt_to_use = test_loader.prompt # Static prompt
            answer_for_eval = answer_string
        
        # Generate completions with Claude
        # Note: generate_claude_completions is async, so we'd ideally run this in an event loop
        # For simplicity in this synchronous-looking structure, we'll await it directly.
        # If num_chains_eval > 1 and you want true parallelism, use asyncio.gather here.
        completions_text = await generate_claude_completions(client, img_path, prompt_to_use, args)
        
        for completion_idx, completion_text in enumerate(completions_text):
            # Process single completion (simplified call, no PDF parts)
            # The 'device' parameter is not relevant for Claude.
            # We pass None for PDF-related args like story, styles, vis_image_path_for_pdf, gui_plotter
            metrics_single = utils._process_single_completion_for_eval(
                completion_text=completion_text, 
                eval_class=eval_class, 
                answer_data=answer_for_eval,
                device="cpu", # Dummy value, not used by evaluator for Claude
                story=None, 
                styles=None, 
                completion_idx=completion_idx,
                dataset_type=args.dataset_type,
                original_image_path=img_path, 
                vis_image_path_for_pdf=None, 
                gui_plotter=None 
            )
            
            num_chains_processed_overall += 1
            if metrics_single and isinstance(metrics_single, dict):
                for metric_name, value in metrics_single.items():
                    # Ensure value is scalar before aggregating
                    val_item = value.item() if hasattr(value, 'item') else value
                    if isinstance(val_item, (int, float)):
                        aggregated_metrics_sum_overall[metric_name] += val_item
                
                if args.dataset_type == 'gui':
                    if is_hard_example:
                        num_chains_processed_hard += 1
                        for metric_name, value in metrics_single.items():
                            val_item = value.item() if hasattr(value, 'item') else value
                            if isinstance(val_item, (int, float)):
                                aggregated_metrics_sum_hard[metric_name] += val_item
                    else:
                        num_chains_processed_normal += 1
                        for metric_name, value in metrics_single.items():
                            val_item = value.item() if hasattr(value, 'item') else value
                            if isinstance(val_item, (int, float)):
                                aggregated_metrics_sum_normal[metric_name] += val_item
        num_examples += 1

    # --- Calculate final averages --- 
    avg_scores_overall = {}
    avg_scores_normal = {}
    avg_scores_hard = {}

    if num_chains_processed_overall > 0:
        for metric_name, total_value in aggregated_metrics_sum_overall.items():
            avg_scores_overall[f"avg_overall_{metric_name}"] = total_value / num_chains_processed_overall 
    
    if args.dataset_type == 'gui':
        if num_chains_processed_normal > 0:
            for metric_name, total_value in aggregated_metrics_sum_normal.items():
                avg_scores_normal[f"avg_normal_{metric_name}"] = total_value / num_chains_processed_normal
        if num_chains_processed_hard > 0:
            for metric_name, total_value in aggregated_metrics_sum_hard.items():
                avg_scores_hard[f"avg_hard_{metric_name}"] = total_value / num_chains_processed_hard
    
    all_avg_scores = {**avg_scores_overall, **avg_scores_normal, **avg_scores_hard}

    # Save to claude_eval.json
    output_json_path = os.path.join(args.output_dir, "claude_eval.json") # Single output file
    try:
        with open(output_json_path, 'w') as f:
            json.dump(all_avg_scores, f, indent=4)
        print(f"Claude evaluation results saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving Claude evaluation JSON: {e}")

    # Determine the main error metric (optional, but good for consistency)
    if args.dataset_type == 'gui':
        main_error_metric_key_for_return = 'metrics/mean_distance_to_center_error'
    elif args.dataset_type == 'clock':
        main_error_metric_key_for_return = 'metrics/mean_abs_error_seconds'
    elif args.dataset_type == 'correlation':
        main_error_metric_key_for_return = 'metrics/mean_abs_correlation_error'
    else:
        main_error_metric_key_for_return = 'reward' # Fallback
        
    final_avg_main_metric_value = avg_scores_overall.get(f"avg_overall_{main_error_metric_key_for_return}", float('nan'))
    print(f"Main error metric (avg_overall_{main_error_metric_key_for_return}): {final_avg_main_metric_value}")
    
    if args.verbose:
        print("\n--- Aggregated Average Scores ---")
        for k, v in all_avg_scores.items():
            print(f"{k}: {v}")

    return all_avg_scores


if __name__ == "__main__":
    args = parse_args()
    
    # Since generate_claude_completions and run_claude_evaluation are async,
    # we need to run them in an asyncio event loop.
    import asyncio
    asyncio.run(run_claude_evaluation(args))
    print("Claude evaluation script finished.")
