"""
Script to run image classification experiments with both CNN and multimodal LLM approaches.
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from flower_dataset import build_flower_dataloaders, NUM_CLASSES
from simpler_grpo.image_classifier import get_model, train_model
from simpler_grpo.utils import seed_everything
from simpler_grpo.datasets import rldatasets


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def run_cnn_experiment(
    dataset_name: str,
    model_name: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 1994,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Run the CNN experiment for image classification.
    
    Args:
        dataset_name: Name of the dataset to use
        model_name: Name of the model architecture to use
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        seed: Random seed for reproducibility
        device: Device to use for training
        
    Returns:
        Tuple of (trained model, training history)
    """
    print(f"Running {model_name} experiment on {dataset_name} dataset...")
    
    # Set random seed
    seed_everything(seed)
    
    # Get dataloaders based on dataset name
    if dataset_name == 'flowers':
        _, _, train_loader, test_loader = build_flower_dataloaders(
            batch_size=batch_size,
            seed=seed
        )
        num_classes = NUM_CLASSES
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: 'flowers'")
    
    # Get model
    model = get_model(model_name, num_classes=num_classes)
    
    # Train the model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Plot training history
    plot_training_history(
        history, 
        save_path=f"plots/{dataset_name}_{model_name}_training.png"
    )
    
    return model, history



def main():
    parser = argparse.ArgumentParser(description='Run image classification experiments')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='flowers',
                      help='Dataset to use (default: flowers)')
    parser.add_argument('--model', type=str, default='resnet50',
                      choices=['simple_cnn', 'resnet50'],
                      help='Model architecture to use (default: simple_cnn)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40,
                      help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1994,
                      help='Random seed (default: 1994)')
    
    # Device argument
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training (default: cuda if available, else cpu)')
    

    
    args = parser.parse_args()
    

    # Run CNN experiment
    run_cnn_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main() 