import torch
import numpy as np
from torch.utils.data import DataLoader
import polars as pl
from pathlib import Path

class AdaptiveBatchSizeStrategy:
    """Base class for adaptive batch size strategies"""
    def __init__(self, min_batch_size=16, max_batch_size=512, initial_batch_size=64):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.previous_loader = None
        
    def get_batch_size(self, **kwargs):
        """Get batch size based on the specific strategy"""
        raise NotImplementedError("Subclasses must implement this method, LOL :))")
        
    def update_loader(self, dataset, **kwargs):
        """Update the dataloader with new batch size"""
        new_batch_size = self.get_batch_size(**kwargs)
        new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
        
        # Only update if batch size changed
        if self.previous_loader is None or self.current_batch_size != new_batch_size:
            self.current_batch_size = new_batch_size
            self.previous_loader = DataLoader(dataset, batch_size=int(self.current_batch_size), shuffle=True)
            
        return self.previous_loader, self.current_batch_size


class LinearBatchSizeStrategy(AdaptiveBatchSizeStrategy):
    """Linearly increases batch size over epochs"""
    def __init__(self, min_batch_size=16, max_batch_size=512, initial_batch_size=64, adjustment_factor=8):
        super().__init__(min_batch_size, max_batch_size, initial_batch_size)
        self.adjustment_factor = adjustment_factor
        
    def get_batch_size(self, epoch_idx=0, **kwargs):
        """Linear adjustment: B_i = B_init + a*i"""
        return self.initial_batch_size + self.adjustment_factor * epoch_idx


class ExponentialBatchSizeStrategy(AdaptiveBatchSizeStrategy):
    """Exponentially increases batch size over epochs"""
    def __init__(self, min_batch_size=16, max_batch_size=512, initial_batch_size=64, base=1.2):
        super().__init__(min_batch_size, max_batch_size, initial_batch_size)
        self.base = base
        
    def get_batch_size(self, epoch_idx=0, **kwargs):
        """Exponential adjustment: B_i = B_init * a^i"""
        return self.initial_batch_size * (self.base ** epoch_idx)


class CyclicalBatchSizeStrategy(AdaptiveBatchSizeStrategy):
    """Varies batch size in cycles"""
    def __init__(self, min_batch_size=16, max_batch_size=512, initial_batch_size=64, cycle_length=10):
        super().__init__(min_batch_size, max_batch_size, initial_batch_size)
        self.cycle_length = cycle_length
        
    def get_batch_size(self, epoch_idx=0, **kwargs):
        """Cyclical adjustment: increase for cycle_length epochs, then restart"""
        cycle_position = epoch_idx % self.cycle_length
        cycle_ratio = cycle_position / self.cycle_length
        range_size = self.max_batch_size - self.min_batch_size
        
        # Triangular cycle: increase and then decrease
        if cycle_position < self.cycle_length / 2:
            return self.min_batch_size + range_size * (cycle_position / (self.cycle_length / 2))
        else:
            return self.max_batch_size - range_size * ((cycle_position - self.cycle_length / 2) / (self.cycle_length / 2))


class LossBasedBatchSizeStrategy(AdaptiveBatchSizeStrategy):
    """Adjusts batch size based on loss changes"""
    def __init__(self, min_batch_size=16, max_batch_size=512, initial_batch_size=64, 
                 adjustment_factor=8, method='simple', alpha=0.3, window_size=5):
        super().__init__(min_batch_size, max_batch_size, initial_batch_size)
        self.adjustment_factor = adjustment_factor
        self.method = method  # 'simple', 'moving_average', or 'variance'
        self.alpha = alpha  # used in moving average method
        self.window_size = window_size  # used in variance method
        self.previous_loss = None
        self.ma_loss = None
        self.loss_history = []
        
    def get_batch_size(self, loss=None, **kwargs):
        """Adjust based on loss changes"""
        if loss is None:
            return self.current_batch_size
            
        loss_delta = 0
        
        # Simple difference
        if self.method == 'simple':
            if self.previous_loss is not None:
                loss_delta = loss - self.previous_loss
            self.previous_loss = loss
                
        # Moving average
        elif self.method == 'moving_average':
            if self.ma_loss is None:
                self.ma_loss = loss
            else:
                old_ma = self.ma_loss
                self.ma_loss = self.alpha * loss + (1 - self.alpha) * self.ma_loss
                loss_delta = loss - old_ma
                
        # Variance over window
        elif self.method == 'variance':
            self.loss_history.append(loss)
            if len(self.loss_history) > self.window_size:
                self.loss_history.pop(0)
                
            if len(self.loss_history) >= 3:  # Need at least a few points for meaningful variance
                mean_loss = sum(self.loss_history) / len(self.loss_history)
                variance = sum((l - mean_loss) ** 2 for l in self.loss_history) / len(self.loss_history)
                # Higher variance means more fluctuation -> decrease batch size
                loss_delta = variance
                
        # If loss is decreasing (negative delta), increase batch size
        adjustment = -self.adjustment_factor * np.sign(loss_delta) if loss_delta != 0 else 0
        return self.current_batch_size + adjustment


class ConfidenceBasedBatchSizeStrategy(AdaptiveBatchSizeStrategy):
    """Adjusts batch size based on model output entropy"""
    def __init__(self, min_batch_size=16, max_batch_size=512, initial_batch_size=64, 
                 adjustment_factor=8):
        super().__init__(min_batch_size, max_batch_size, initial_batch_size)
        self.adjustment_factor = adjustment_factor
        self.previous_entropy = None
        self.entropies = []
    
    def update(self, outputs=None, **kwargs):
        """Accumulate outputs for entropy calculation"""
        if outputs is not None:
            if not hasattr(self, 'entropy_buffer'):
                self.entropy_buffer = []
            self.entropy_buffer.append(self.compute_entropy(outputs))

    def compute_entropy(self, outputs):
        """Compute average entropy from model outputs"""
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        # Calculate entropy: -sum(p_i * log(p_i))
        log_probs = torch.log(probs + 1e-10)  # Add small constant to avoid log(0)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy.mean().item()
        
    def get_batch_size(self, **kwargs):
        """Adjust based on output entropy averaged over the epoch"""
        # Get accumulated entropy from updates
        if not hasattr(self, 'entropy_buffer'):
            return self.current_batch_size
            
        # Calculate average entropy for the epoch
        avg_entropy = np.mean(self.entropy_buffer) if self.entropy_buffer else 0
        self.entropy_buffer.clear()  # Reset for next epoch

        # Initial condition
        if self.previous_entropy is None:
            self.previous_entropy = avg_entropy
            return self.current_batch_size

        # Calculate entropy change
        delta = avg_entropy - self.previous_entropy
        self.previous_entropy = avg_entropy

        # Proportional adjustment based on entropy change
        adjustment = -self.adjustment_factor * np.sign(delta)
        
        # Apply adjustment with bounds checking
        new_size = self.current_batch_size + adjustment
        new_size = max(self.min_batch_size, min(self.max_batch_size, new_size))
        
        return new_size


class GNSBatchSizeStrategy(AdaptiveBatchSizeStrategy):
    """Adjusts batch size based on Gradient Noise Scale (GNS)"""
    def __init__(self, min_batch_size=16, max_batch_size=512, initial_batch_size=64, 
                 adjustment_factor=2, num_microbatches=4, noise_threshold=1.0):
        super().__init__(min_batch_size, max_batch_size, initial_batch_size)
        self.adjustment_factor = adjustment_factor
        self.num_microbatches = num_microbatches
        self.previous_gns = None
        self.noise_threshold = noise_threshold
        self.gns_history = []
        self.history_size = 3  # Number of GNS values to keep for smoothing
        
    def compute_gns(self, model, loss_func, X, y):
        """
        Compute Gradient Noise Scale: the ratio of gradient variance to squared gradient norm
        GNS = Trace(Σ) / ||E[g]||^2 where:
        - Σ is the covariance matrix of gradients
        - E[g] is the expected gradient (mean)
        """
        # Split batch into microbatches
        batch_size = X.size(0)
        microbatch_size = max(1, batch_size // self.num_microbatches)
        
        # Store gradients for each microbatch
        gradients = []
        
        for i in range(0, batch_size, microbatch_size):
            end_idx = min(i + microbatch_size, batch_size)
            X_micro, y_micro = X[i:end_idx], y[i:end_idx]
            
            # Zero gradients
            model.zero_grad()
            
            # Forward and backward pass
            output = model(X_micro)
            loss = loss_func(output, y_micro)
            loss.backward()
            
            # Extract and flatten gradients
            grad = []
            for param in model.parameters():
                if param.grad is not None:
                    grad.append(param.grad.data.view(-1).clone())  # Clone to avoid in-place modifications
            
            if grad:
                grad = torch.cat(grad)
                gradients.append(grad)
        
        if not gradients or len(gradients) < 2:
            return self.previous_gns if self.previous_gns is not None else 1.0
        
        # Stack gradients for vectorized computation
        grad_tensor = torch.stack(gradients)
        
        # Compute mean gradient vector across microbatches
        mean_grad = torch.mean(grad_tensor, dim=0)
        
        # Compute trace of the covariance matrix (sum of gradient variances)
        # This is more efficient than computing the full covariance matrix
        grad_var = torch.mean(torch.var(grad_tensor, dim=0))
        
        # Compute squared norm of mean gradient
        mean_grad_norm_squared = torch.sum(mean_grad**2)
        
        # Avoid division by zero
        if mean_grad_norm_squared < 1e-8:
            return self.previous_gns if self.previous_gns is not None else 1.0
        
        # Compute GNS as the ratio of gradient variance to squared gradient norm
        gns = grad_var / mean_grad_norm_squared
        
        return gns.item()
        
    def get_batch_size(self, model=None, loss_func=None, X=None, y=None, **kwargs):
        """Adjust batch size based on Gradient Noise Scale"""
        if model is None or loss_func is None or X is None or y is None:
            return self.current_batch_size
            
        # Compute current GNS
        gns = self.compute_gns(model, loss_func, X, y)
        
        # Add to history and keep limited size
        if not hasattr(self, 'gns_history'):
            self.gns_history = []
        self.gns_history.append(gns)
        if len(self.gns_history) > self.history_size:
            self.gns_history.pop(0)
            
        # Use smoothed GNS (average over recent history)
        smoothed_gns = sum(self.gns_history) / len(self.gns_history)
        
        # Print GNS values to debug
        print(f"Current GNS: {gns:.4f}, Smoothed GNS: {smoothed_gns:.4f}, Threshold: {self.noise_threshold}")
        
        # Adjust batch size based on smoothed GNS
        new_batch_size = self.current_batch_size
        if smoothed_gns > self.noise_threshold:
            # High gradient noise - increase batch size
            new_batch_size = int(self.current_batch_size * self.adjustment_factor)
            print(f"High noise detected. Increasing batch size from {self.current_batch_size} to {new_batch_size}")
        elif smoothed_gns < self.noise_threshold * 0.5:  # Add hysteresis to prevent oscillation
            # Low noise detected. Decreasing batch size
            new_batch_size = max(int(self.current_batch_size / self.adjustment_factor), self.min_batch_size)
            print(f"Low noise detected. Decreasing batch size from {self.current_batch_size} to {new_batch_size}")
        
        # Return the new batch size (will be bounded by min/max in the update_loader method)
        return new_batch_size


def create_results_dir():
    """Create results directory if it doesn't exist"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir


def save_training_data(experiment_name, epochs_data):
    """Save training data to CSV using Polars"""
    results_dir = create_results_dir()
    filename = results_dir / f"ex{experiment_name}.csv"
    
    # Convert data to Polars DataFrame
    df = pl.DataFrame(epochs_data)
    
    # Save to CSV
    df.write_csv(filename)
    
    return filename