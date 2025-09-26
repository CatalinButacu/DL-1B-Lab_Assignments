import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import copy
from tqdm import tqdm
from batching import ConfidenceBasedBatchSizeStrategy

def train_model(model, dataset, test_dataset, num_epochs=100, learn_rate=0.01, 
                batch_strategy=None, experiment_name="default"):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    # Initial batch size setup
    batch_size = 64 if batch_strategy is None else batch_strategy.current_batch_size
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # For tracking metrics
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
        'batch_sizes': []
    }
    
    # To track best model
    best_test_acc = 0.0
    best_model = None
    
    # Training loop
    print(f"\nStarting training with {batch_strategy.__class__.__name__} strategy")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Adjust batch size if we have a strategy
        if batch_strategy is not None:
            # Get batch size for this epoch
            batch_size = batch_strategy.get_batch_size(
                epoch_idx=epoch,
                loss=metrics['train_loss'][-1] if metrics['train_loss'] else None,
            )
            train_loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
            
        # Record the batch size
        metrics['batch_sizes'].append(batch_size)
        
        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Batch: {batch_size}]")
        
        # Process batches
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # For GNS calculation (if using GNSBatchSizeStrategy)
            microbatch_gradients = []
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            if batch_strategy is not None and isinstance(batch_strategy, ConfidenceBasedBatchSizeStrategy):
                batch_strategy.update(outputs=outputs.detach())
            
            # Collect gradients for metrics if using GNS
            if hasattr(batch_strategy, 'num_microbatches'):
                # Store gradient values before optimizer step
                grad_dict = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_dict[name] = param.grad.clone().detach()
                microbatch_gradients.append(grad_dict)
                
                # Update batch size based on gradient info
                if batch_strategy is not None and hasattr(batch_strategy, 'update'):
                    batch_strategy.update(
                        epoch_idx=epoch,
                        microbatch_gradients=microbatch_gradients
                    )
                
            optimizer.step()
            
            # Track metrics within this epoch
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == targets).sum().item()
            correct += batch_correct
            total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': f"{100.0 * batch_correct / targets.size(0):.2f}%"
            })
            
            # Update batch strategy with prediction confidence info
            if batch_strategy is not None and hasattr(batch_strategy, 'update'):
                batch_strategy.update(
                    epoch_idx=epoch,
                    loss=loss.item(),
                    targets=targets
                )
                
        # Compute epoch metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Store metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
              f"Batch Size: {batch_size}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = copy.deepcopy(model)
            
    # Save metrics to file
    save_path = save_metrics(metrics, experiment_name)
    print(f"Training completed. Metrics saved to {save_path}")
    
    # Plot training curves
    plot_learning_curves(metrics, experiment_name)
    plot_batch_size_history(metrics, experiment_name)
    
    # Return best model and metrics
    return best_model, metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def save_metrics(metrics, experiment_name):
    """Save metrics to CSV file"""
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Prepare save path
    save_path = results_dir / f"{experiment_name}_metrics.npz"
    
    # Save metrics as numpy compressed file
    np.savez(
        save_path,
        train_loss=np.array(metrics['train_loss']),
        test_loss=np.array(metrics['test_loss']),
        train_acc=np.array(metrics['train_acc']),
        test_acc=np.array(metrics['test_acc']),
        batch_sizes=np.array(metrics['batch_sizes'])
    )
    
    return save_path


def plot_learning_curves(metrics, experiment_name):
    """Plot training and test learning curves"""
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot loss curves
    ax1.plot(metrics['train_loss'], label='Train Loss')
    ax1.plot(metrics['test_loss'], label='Test Loss')
    ax1.set_title(f'Loss Curves - {experiment_name}')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    ax2.plot(metrics['train_acc'], label='Train Acc')
    ax2.plot(metrics['test_acc'], label='Test Acc')
    ax2.set_title(f'Accuracy Curves - {experiment_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(results_dir / f"{experiment_name}_learning_curves.png")
    plt.close()


def plot_batch_size_history(metrics, experiment_name):
    """Plot batch size evolution"""
    # Ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Plot batch size history
    plt.plot(metrics['batch_sizes'], marker='o')
    plt.title(f'Batch Size Evolution - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Size')
    plt.grid(True)
    
    # Add horizontal lines for min/max if available
    if len(metrics['batch_sizes']) > 0:
        min_batch = min(metrics['batch_sizes'])
        max_batch = max(metrics['batch_sizes'])
        plt.axhline(min_batch, color='r', linestyle='--', alpha=0.5, 
                   label=f'Min: {min_batch}')
        plt.axhline(max_batch, color='g', linestyle='--', alpha=0.5,
                   label=f'Max: {max_batch}')
        plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(results_dir / f"{experiment_name}_batch_sizes.png")
    plt.close()