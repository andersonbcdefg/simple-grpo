# Simple GRPO Implementation Guide

This document provides a comprehensive overview of the Simple GRPO (Group Relative Policy Optimization) implementation for vision-language models.

## Table of Contents
1. [Overview](#overview)
2. [Key Components](#key-components)
3. [GRPO Training Process](#grpo-training-process)
4. [Important Parameters](#important-parameters)
5. [Memory Considerations](#memory-considerations)
6. [Dataset Types](#dataset-types)
7. [File Structure](#file-structure)

## Overview

This repository implements GRPO (DeepSeek-style) training for vision-language models, specifically targeting Qwen-VL. The implementation supports multiple visual reasoning tasks including clock reading, correlation estimation, GUI interaction, and CAPTCHA solving.

## Key Components

### Main Training Loop (`main.py`)
- **Location**: Lines 940-1201
- **Key functions**:
  - `grpo_loss()`: Orchestrates the GRPO loss computation (lines 510-603)
  - `generate_completions()`: Generates multiple completion chains (lines 228-351)
  - `score_completions()`: Evaluates completions and computes advantages (lines 353-445)
  - `compute_loss()`: Calculates the GRPO loss with advantages (lines 447-508)

### GRPO Algorithm Steps

1. **Generation Phase** (`generate_completions`, lines 228-351):
   - Takes an image and prompt
   - Generates `num_chains` completions in parallel (default: 16)
   - All chains generate until EOS or `max_completion_length`
   - Handles variable-length completions with padding

2. **Scoring Phase** (`score_completions`, lines 353-445):
   - Evaluates each completion using task-specific evaluators
   - Computes rewards for each completion
   - Calculates advantages using mean and std of rewards within the group
   - Formula: `advantages = (rewards - mean) / (std + 1e-4)`

3. **Loss Computation** (`compute_loss`, lines 447-508):
   - Recomputes log probabilities for generated tokens
   - **Critical**: Calls `get_per_token_logps_vl()` which reprocesses images
   - Applies GRPO loss: `-exp(logp - logp.detach()) * advantages`
   - No KL penalty in this implementation

4. **Gradient Accumulation** (lines 1009-1017):
   - Accumulates gradients over `gradient_accumulation_steps` (default: 4)
   - Only calls `optimizer.step()` every N steps
   - **Memory bottleneck**: All gradients accumulate in GPU memory

## Important Parameters

### Training Configuration
```python
--num_train_iters: 3000          # Total training iterations
--gradient_accumulation_steps: 4  # Steps before optimizer update
--learning_rate: 5e-6            # AdamW learning rate
--adam_beta1: 0.9               # Adam beta1
--adam_beta2: 0.99              # Adam beta2  
--weight_decay: 0.1             # Weight decay
--max_grad_norm: 0.1            # Gradient clipping threshold
--warmup_percent: 0.18          # Warmup percentage of total steps
```

### Generation Parameters
```python
--num_chains: 16                 # Parallel completion chains (training)
--num_chains_eval: 2             # Parallel chains for evaluation
--max_completion_length: 786     # Maximum tokens per completion
--temperature: 0.9               # Sampling temperature
```

### Memory-Critical Parameters
```python
--num_chains: 16                 # Directly impacts activation memory
--max_completion_length: 786     # Worst-case memory allocation
--gradient_accumulation_steps: 4 # Delays memory release
```

## Memory Considerations

### GPU Memory Breakdown (7B model, bf16)
1. **Model weights**: 14GB (7B × 2 bytes)
2. **Gradients**: 14GB (7B × 2 bytes)
3. **AdamW optimizer states**: 56GB (momentum + variance in fp32)
   - Momentum: 28GB (7B × 4 bytes)
   - Variance: 28GB (7B × 4 bytes)
4. **Total static**: ~84GB (exceeds 80GB GPU!)

### Variable Memory Usage
- **Activations**: Scale with `batch_size × seq_length × hidden_dim`
- **Attention**: O(batch_size × seq_length² × num_heads) per layer
- **Critical issue**: All chains padded to longest completion in batch

### OOM Triggers
1. **Variable completion lengths**: One long completion (e.g., 786 tokens) forces all 16 chains to that length
2. **Gradient accumulation**: Activations accumulate over 4 steps before release
3. **Image reprocessing**: `get_per_token_logps_vl()` duplicates image processing

### Memory Optimization Strategies
1. Reduce `num_chains` (16 → 8)
2. Reduce `max_completion_length` (786 → 400)
3. Use 8-bit AdamW optimizer
4. Add `torch.cuda.empty_cache()` after backward passes
5. Consider gradient checkpointing

## Dataset Types

### 1. Clock (`clock`)
- **Task**: Read analog clock time
- **Loader**: `ClockDataLoader` 
- **Evaluator**: `ClockEvaluator`
- **Rewards**: Correctness (±3), time format (0.5), XML format (0.5)

### 2. Correlation (`correlation`)
- **Task**: Estimate Pearson correlation from scatter plot
- **Loader**: `CorrelationScatterDataLoader`
- **Evaluator**: `CorrelationEvaluator`
- **Rewards**: Correctness (0-1), format (0.5), XML format (0.5)

### 3. GUI (`gui`)
- **Task**: Click on specified GUI elements
- **Loader**: `GUIDataLoader`
- **Evaluator**: `GUIEvaluator`
- **Rewards**: XML format (0.5), click hit (3), distance to center (±2)
- **Special**: Dynamic prompts, hard mode support

### 4. CAPTCHA (`captcha`)
- **Task**: Select squares containing target objects
- **Loader**: `PreGeneratedCaptchaLoader`
- **Evaluator**: `CaptchaEvaluator`
- **Rewards**: F1 score (0-1)
- **Note**: Requires pre-generated dataset

## File Structure

### Core Training
- `main.py`: Main training loop and GRPO implementation
- `src/simple_grpo/utils/grpo.py`: Log probability computation
- `src/simple_grpo/evaluator.py`: Task-specific reward evaluators

### Datasets
- `src/simple_grpo/datasets/__init__.py`: Data loaders for all tasks
- `src/simple_grpo/datasets/*_generator.py`: Image generation for each task

### Utilities
- `src/simple_grpo/llms.py`: Model loading and configuration
- `src/simple_grpo/prompt.py`: Prompt creation utilities
- `src/simple_grpo/utils/logging.py`: Training and evaluation logging
- `src/simple_grpo/utils/reports.py`: PDF report generation

## Critical Implementation Details

### Image Path Changes
- **Recent change**: Static filenames → unique UUIDs
- **Impact**: Prevents image caching, increases memory pressure
- **Files never cleaned up**: Temporary images accumulate on disk

### Vision Processing Duplication
- `generate_completions()` processes images once
- `get_per_token_logps_vl()` reprocesses the same images
- Consider caching processed vision features

### Gradient Accumulation Pitfall
- All chains must complete before any gradient step
- Memory peaks at `num_chains × max_completion_length × gradient_accumulation_steps`
- No intermediate memory release between accumulation steps

## Running the Code

### Basic Training
```bash
python main.py --dataset_type clock --output_dir output/clock_experiment
```

### Memory-Optimized Training
```bash
python main.py --dataset_type clock --num_chains 8 --max_completion_length 400 --gradient_accumulation_steps 2
```

### Evaluation Only
```bash
python main.py --dataset_type clock --num_train_iters 0 --eval_iterations 1
```

## Common Issues and Solutions

### CUDA OOM Errors
1. **Symptom**: OOM during `optimizer.step()` after gradient accumulation
2. **Cause**: Memory usage scales with longest completion × num_chains
3. **Solutions**:
   - Reduce `num_chains` and/or `max_completion_length`
   - Use 8-bit optimizer
   - Reduce `gradient_accumulation_steps`

### Slow Training
1. **Image reprocessing**: Consider caching vision features
2. **PDF generation**: Happens every `save_steps` (default: 3000)
3. **Evaluation**: Runs every `eval_iterations` (default: 50)

### Dataset Issues
1. **Temporary files**: Clean up old temp_*.png files periodically
2. **CAPTCHA dataset**: Must run `captcha_ds.py` first
3. **GUI hard mode**: Control with `hard_mode_prob` parameter