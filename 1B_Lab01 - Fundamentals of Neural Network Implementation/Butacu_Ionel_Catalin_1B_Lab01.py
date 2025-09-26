import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import os

torch.manual_seed(42)
np.random.seed(42)


class ActivationType(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"

class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"

class LossType(Enum):
    MSE = "mse"
    MAE = "mae"
    BCE = "bce"


@dataclass
class ModelConfig:
    input_size: int
    hidden_layers: List[int]
    activation: ActivationType
    loss_type: LossType
    optimizer_type: OptimizerType
    learning_rate: float
    is_classification: bool = False

class FlexibleNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super(FlexibleNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = config.input_size
        for hidden_size in config.hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(nn.Linear(prev_size, 1))
        
        # Set activation function
        self.activation = {
            ActivationType.RELU: nn.ReLU(),
            ActivationType.LEAKY_RELU: nn.LeakyReLU(),
            ActivationType.SIGMOID: nn.Sigmoid(),
            ActivationType.TANH: nn.Tanh()
        }[config.activation]
        
        self.final_activation = nn.Sigmoid() if config.is_classification else nn.Identity()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return self.final_activation(x)

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, current_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = current_loss
            self.counter = 0
        return self.should_stop


def generate_dataset(
    num_features: int,
    num_instances: int,
    noise_level: float = 0,
    is_classification: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Generate synthetic dataset 
    """
    X = torch.randn(num_instances, num_features)
    
    if is_classification:
        weights = torch.rand(num_features, 1)
        logits = X @ weights + 0.5
        probabilities = torch.sigmoid(logits)
        y = (probabilities > 0.5).float()
    else:
        weights = torch.rand(num_features, 1)
        bias = 0.5
        y = X @ weights + bias + noise_level * torch.randn(num_instances, 1)
    
    return X, y

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate R^2 score
    """
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

def calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate classification accuracy
    """
    predictions = (y_pred > 0.5).float()
    return (predictions == y_true).float().mean().item()

def train_model(
    model: nn.Module,
    config: ModelConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_epochs: int = 100,
    early_stopping: Optional[EarlyStopping] = None
) -> Dict:
    """
    Train the model and return training history
    """
    loss_functions = {
        LossType.MSE: nn.MSELoss(),
        LossType.MAE: nn.L1Loss(),
        LossType.BCE: nn.BCELoss()
    }
    loss_func = loss_functions[config.loss_type]
    
    optimizers = {
        OptimizerType.ADAM: optim.Adam,
        OptimizerType.SGD: optim.SGD,
        OptimizerType.RMSPROP: optim.RMSprop,
        OptimizerType.ADAGRAD: optim.Adagrad
    }
    optimizer = optimizers[config.optimizer_type](model.parameters(), lr=config.learning_rate)
    
    history = {
        'train_loss': [], 'test_loss': [],
        'train_r2': [], 'test_r2': [],
        'train_acc': [], 'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = loss_func(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = loss_func(test_outputs, y_test)
            
            # Calculate metrics
            if config.is_classification:
                train_acc = calculate_accuracy(train_outputs, y_train)
                test_acc = calculate_accuracy(test_outputs, y_test)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)
            else:
                train_r2 = r2_score(y_train, train_outputs)
                test_r2 = r2_score(y_test, test_outputs)
                history['train_r2'].append(train_r2)
                history['test_r2'].append(test_r2)
            
            history['train_loss'].append(train_loss.item())
            history['test_loss'].append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}:')
            print(f'  Train - Loss: {train_loss.item():.4f}', end='')
            print(f', {"Acc" if config.is_classification else "R^2"}: '
                  f'{(train_acc if config.is_classification else train_r2):.4f}')
            print(f'  Test  - Loss: {test_loss.item():.4f}', end='')
            print(f', {"Acc" if config.is_classification else "R^2"}: '
                  f'{(test_acc if config.is_classification else test_r2):.4f}')
        
        if early_stopping and early_stopping(test_loss.item()):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    return history

def plot_training_history(histories: Dict[str, Dict], is_classification: bool = False, experiment_name: str = "default"):
    """
    Simple plot & save metrics training
    """
    filename = experiment_name.lower().replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange']
    
    for (name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, history['train_loss'], f'{color}-', label=f'{name} (Train Loss)')
        ax1.plot(epochs, history['test_loss'], f'{color}--', label=f'{name} (Test Loss)')
        
        # Plot R^2/Accuracy
        metric_key = 'train_acc' if is_classification else 'train_r2'
        metric_name = 'Accuracy' if is_classification else 'R^2 Score'
        
        ax2.plot(epochs, history[metric_key], f'{color}-', label=f'{name} (Train {metric_name})')
        ax2.plot(epochs, history[metric_key.replace('train', 'test')], f'{color}--', label=f'{name} (Test {metric_name})')
    
    ax1.set_title(f'Loss over Epochs - {experiment_name}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title(f'{metric_name} over Epochs - {experiment_name}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = f'results/training_progress_{filename}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training progress plot saved as '{save_path}'")


def init_data(config: ModelConfig, noise_level: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize the data for the experiments
    """
    X, y = generate_dataset(config.input_size, num_instances=100, noise_level=noise_level)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test


def main():
    os.makedirs('results', exist_ok=True)
    
    print("Task 1 - Base model")
    base_config = ModelConfig(
        input_size=3,
        hidden_layers=[6, 3],
        activation=ActivationType.RELU,
        loss_type=LossType.MSE,
        optimizer_type=OptimizerType.ADAM,
        learning_rate=0.01
    )

    X_train, y_train, X_test, y_test = init_data(config=base_config,noise_level=0)

    model = FlexibleNN(base_config)
    base_history = train_model(model, base_config, X_train, y_train, X_test, y_test)
    plot_training_history({'Base Model': base_history}, is_classification=False, experiment_name="Task1_Base_Model_Regression")



    print("Task 2 - Increased features")
    print("\tA - 10 features used")
    base_config.input_size = 10

    X_train, y_train, X_test, y_test = init_data(config=base_config,noise_level=0)

    model = FlexibleNN(base_config)
    more_features_history = train_model(model, base_config, X_train, y_train, X_test, y_test)
    plot_training_history({'Increased Features': more_features_history}, is_classification=False, experiment_name="Task2_Increased_Features_from_3_to_10")
    
    print("\tB - noise added")
    X_train, y_train, X_test, y_test = init_data(config=base_config,noise_level=0.1)

    model = FlexibleNN(base_config)
    noise_features_history = train_model(model, base_config, X_train, y_train, X_test, y_test)
    plot_training_history({'Noise Added': noise_features_history}, is_classification=False, experiment_name="Task2_Noise_Added")
    


    print("Task 3 - Different activation functions")    
    activation_histories = {}
    for activation in [ActivationType.RELU, 
                       ActivationType.LEAKY_RELU, 
                       ActivationType.SIGMOID, 
                       ActivationType.TANH]:
        config = ModelConfig(**base_config.__dict__)
        config.activation = activation
        X_train, y_train, X_test, y_test = init_data(config=config,noise_level=0.1)
        model = FlexibleNN(config)
        history = train_model(model, config, X_train, y_train, X_test, y_test)
        activation_histories[activation.value] = history
    
    plot_training_history(activation_histories, False, "Task3_Exploring Activation Functions")
   


    print("Task 4 - Modify the Loss Function")
    loss_histories = {}
    for loss in [LossType.MSE, LossType.MAE]:
        config = ModelConfig(**base_config.__dict__)
        config.loss_type = loss
        X_train, y_train, X_test, y_test = init_data(config=config,noise_level=0.1)
        model = FlexibleNN(config)
        history = train_model(model, config, X_train, y_train, X_test, y_test)
        loss_histories[loss.value] = history

    plot_training_history(loss_histories, False, "Task4_Exploring Loss Functions")


    print("Task 5 - Change the optimizer")
    optimizer_histories = {}
    for optimizer in [OptimizerType.SGD, 
                      OptimizerType.RMSPROP, 
                      OptimizerType.ADAGRAD,
                      OptimizerType.ADAM]:
        config = ModelConfig(**base_config.__dict__)
        config.optimizer_type = optimizer
        X_train, y_train, X_test, y_test = init_data(config=config,noise_level=0.1)
        model = FlexibleNN(config)
        history = train_model(model, config, X_train, y_train, X_test, y_test)
        optimizer_histories[optimizer.value] = history
    
    plot_training_history(optimizer_histories, False, "Task5_Exploring Optimizers")



    print("Task 6 - Change the network architecture")
    architecture_histories = {}
    for hidden_layers in [[6, 3], [10, 5], [20, 10]]:
        config = ModelConfig(**base_config.__dict__)
        config.hidden_layers = hidden_layers
        X_train, y_train, X_test, y_test = init_data(config=config,noise_level=0.1)
        model = FlexibleNN(config)
        history = train_model(model, config, X_train, y_train, X_test, y_test)
        architecture_histories[str(hidden_layers)] = history

    plot_training_history(architecture_histories, False, "Task6_Exploring Network Architectures")



    print("Task 7 - Early stopping")
    config = ModelConfig(**base_config.__dict__)
    early_stopping_histories = {}

    X_train, y_train, X_test, y_test = init_data(config=config,noise_level=0.1)
    model_no_early_stopping = FlexibleNN(config)
    early_stopping_histories['No Early Stopping'] = train_model(model_no_early_stopping, config, X_train, y_train, X_test, y_test)

    X_train, y_train, X_test, y_test = init_data(config=config,noise_level=0.1)
    early_stopping = EarlyStopping(patience=10)
    model_with_early_stopping = FlexibleNN(config)
    early_stopping_histories['With Early Stopping'] = train_model(model_with_early_stopping, config, X_train, y_train, X_test, y_test)
    plot_training_history(early_stopping_histories, False, "Task7_Early_Stopping")


    print("Task 8 - Binary Classification")
    config = ModelConfig(**base_config.__dict__)
    config.is_classification = True
    model = FlexibleNN(config)
    X, y = generate_dataset(config.input_size, num_instances=100, is_classification=True)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    classification_history = train_model(model, config, X_train, y_train, X_test, y_test)
    plot_training_history({'Binary Classification': classification_history}, is_classification=True, experiment_name="Task8_Binary_Classification")


    
if __name__ == "__main__":
    main()