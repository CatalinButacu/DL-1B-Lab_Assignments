import sys
import torch
import torch.nn as nn 
import pandas as pd
import os

from models import IrisNet, FashionCNN, CustomNet
from training import train_model
from datasets import load_iris_dataset, load_fashion_mnist_dataset, load_custom_dataset
from batching import (
    LinearBatchSizeStrategy, 
    ExponentialBatchSizeStrategy,
    CyclicalBatchSizeStrategy,
    LossBasedBatchSizeStrategy,
    ConfidenceBasedBatchSizeStrategy,
    GNSBatchSizeStrategy,
    create_results_dir
)

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

torch.manual_seed(42)


def run_task(task_id, model, train_dataset, test_dataset, batch_strategy, num_epochs=50, learn_rate=0.01):
    """Run a task with a specific adaptive batching strategy"""
    print(f"\n{'='*80}\nTask {task_id}: {batch_strategy.__class__.__name__}\n{'='*80}")
    
    # Clone the model to start fresh
    if isinstance(model, nn.Module):
        # Create a new instance of the same class
        if isinstance(model, IrisNet):
            model_clone = IrisNet()
        elif isinstance(model, FashionCNN):
            model_clone = FashionCNN()
        elif isinstance(model, CustomNet):
            num_features = model.fc1.in_features
            num_classes = model.fc2.out_features
            model_clone = CustomNet(num_features, num_classes)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    else:
        raise ValueError("Model must be a PyTorch module")
    
    # Train with the adaptive strategy
    trained_model, metrics = train_model(
        model=model_clone,
        dataset=train_dataset,
        test_dataset=test_dataset,
        num_epochs=num_epochs,
        learn_rate=learn_rate,
        batch_strategy=batch_strategy,
        experiment_name=task_id
    )
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create a DataFrame with all metrics
    df = pd.DataFrame({
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': metrics['train_losses'],
        'test_loss': metrics['test_losses'],
        'train_accuracy': metrics['train_accuracies'],
        'test_accuracy': metrics['test_accuracies'],
        'batch_size': batch_strategy.stats['batch_sizes']
    })
    
    # Save to CSV
    csv_path = f"results/{task_id}_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    return trained_model, metrics

def main():    
    # Create results directory
    create_results_dir()
    
    # Hyper-parameters
    num_epochs = 100
    learn_rate = 0.01
    min_batch = 16
    max_batch = 512
    init_batch = 64
    
    # LOADING DATASET
    print("\nLoading IRIS dataset...")
    iris_train, iris_test = load_iris_dataset()
    iris_model = IrisNet()

    print("\nLoading FASHION MINST dataset...")
    fashion_train, fashion_test = load_fashion_mnist_dataset()
    fashion_model = FashionCNN()

    print("\nLoading CustomNet dataset...")
    num_features = 16
    num_classes = 8
    custom_train, custom_test = load_custom_dataset(
                                                    num_instances=2000, 
                                                    num_features=num_features, 
                                                    num_classes=num_classes, 
                                                    class_separation=1.5,
                                                    noise=0.01
                                                )
    custom_model = CustomNet(num_features, num_classes)

    # List of tasks to run
    tasks = []
    
    
    
    print("\n===== EXERCISE 6: GRADIENT NOISE SCALE BATCH ADAPTATION =====")
    
    tasks.append({
        "id": "6.1_iris_gns",
        "exercise": 6,
        "model": iris_model,
        "train_dataset": iris_train,
        "test_dataset": iris_test,
        "batch_strategy": GNSBatchSizeStrategy(
            min_batch, max_batch, init_batch, 
            adjustment_factor=12, num_microbatches=3
        ),
        "description": "Gradient noise scale batch adjustment on Iris dataset"
    })
    
    tasks.append({
        "id": "6.2_fashion_gns",
        "exercise": 6,
        "model": fashion_model,
        "train_dataset": fashion_train,
        "test_dataset": fashion_test,
        "batch_strategy": GNSBatchSizeStrategy(
            min_batch, max_batch, init_batch, 
            adjustment_factor=20, num_microbatches=5
        ),
        "description": "Gradient noise scale batch adjustment on Fashion MNIST dataset"
    })
    
    tasks.append({
        "id": "6.3_custom_gns",
        "exercise": 6,
        "model": custom_model,
        "train_dataset": custom_train,
        "test_dataset": custom_test,
        "batch_strategy": GNSBatchSizeStrategy(
            min_batch, max_batch, init_batch, 
            adjustment_factor=16, num_microbatches=4
        ),
        "description": "Gradient noise scale batch adjustment on custom dataset"
    })


    # Run all tasks
    results = {}
    
    # Sort tasks by exercise and ID
    sorted_tasks = sorted(tasks, key=lambda x: (x["exercise"], x["id"]))
    
    # Run tasks by exercise group
    current_exercise = None
    for task in sorted_tasks:
        if current_exercise != task["exercise"]:
            current_exercise = task["exercise"]
            print(f"\n\n{'='*80}\nRUNNING EXERCISE {current_exercise} TASKS\n{'='*80}")
        
        try:
            print(f"\n--- Task {task['id']}: {task['description']} ---")
            model, metrics = run_task(
                task_id=task["id"],
                model=task["model"],
                train_dataset=task["train_dataset"],
                test_dataset=task["test_dataset"],
                batch_strategy=task["batch_strategy"],
                num_epochs=num_epochs,
                learn_rate=learn_rate
            )
            results[task["id"]] = metrics
        except Exception as e:
            print(f"Error in task {task['id']}: {e}")
    
    print("\nAll tasks completed mate!")
    print(f"All results saved in the 'results' directory.")


if __name__ == "__main__":
    main()