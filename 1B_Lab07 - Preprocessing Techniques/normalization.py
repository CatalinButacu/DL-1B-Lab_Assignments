import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

torch.manual_seed(0)

def gen_dataset():
    x1 = torch.randint(1000, 5000, (1000, 1), dtype = torch.float32)
    x2 = torch.rand(1000, 1) * 0.8 + 0.1
    X = torch.cat([x1, x2], dim=1)

    y = 0.001 * x1 + 10 * x2 + torch.randn(1000, 1) * 0.5

    return X, y

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(), 
            nn.Linear(16, 1)
            )

    def forward(self, x):
        return self.model(x)

    def train_model(self, X_train, y_train, X_test, y_test, learn_rate = 0.01, num_epochs = 10000, accuracy_threshold = 0.9):
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learn_rate)

        losses = []
        test_accuracies = []
        stop_epoch = None

        for ep in range(num_epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = loss_func(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Calculate test accuracy (R-squared)
            with torch.no_grad():
                test_outputs = self.forward(X_test)
                ss_res = torch.sum((y_test - test_outputs) ** 2)
                ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                test_accuracies.append(r_squared.item())

            losses.append(loss.item())
            #print(f'Epoch: {ep}, Loss: {loss.item():.4f}\nTest R²: {r_squared.item():.4f}')
            
            # Check for early stopping based on accuracy
            if test_accuracies[-1] >= accuracy_threshold and stop_epoch is None:
                stop_epoch = ep
                print(f'EARLY STOP TRAINING: Accuracy: {test_accuracies[-1]}, Loss: {loss.item():.4f} at Epoch: {ep}')
                break
        
        return losses, test_accuracies, stop_epoch

if __name__ == '__main__':
    
    X, y = gen_dataset()
    
    # Split into train/test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Original data
    print('Training model with raw data...')
    model_raw = RegressionNet()
    losses_raw, test_acc_raw, _ = model_raw.train_model(X_train, y_train, X_test, y_test)

    # Standardized data
    print('Training model with standardized data...')
    scaler = StandardScaler()
    X_std_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_std_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    model_std = RegressionNet()
    losses_std, test_acc_std, _ = model_std.train_model(X_std_train, y_train, X_std_test, y_test)

    # Min-max scaled data
    print('Training model with min-max scaled data...')
    minmax = MinMaxScaler()
    X_minmax_train = torch.tensor(minmax.fit_transform(X_train), dtype=torch.float32)
    X_minmax_test = torch.tensor(minmax.transform(X_test), dtype=torch.float32)
    model_minmax = RegressionNet()
    losses_minmax, test_acc_minmax, _ = model_minmax.train_model(X_minmax_train, y_train, X_minmax_test, y_test)

    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    # Loss plots
    plt.subplot(2, 3, 1)
    plt.plot(np.array(losses_raw), label = 'raw', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 3, 2)
    plt.plot(np.array(losses_std), label = 'standardized', color='green')
    plt.xlabel('epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 3, 3)
    plt.plot(np.array(losses_minmax), label = 'min-max scaled', color='red')
    plt.xlabel('epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid()
    
    # Test accuracy plots
    plt.subplot(2, 3, 4)
    plt.plot(np.array(test_acc_raw), label='raw', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('Test R²')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 3, 5)
    plt.plot(np.array(test_acc_std), label='standardized', color='green')
    plt.xlabel('epoch')
    plt.ylabel('Test R²')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 3, 6)
    plt.plot(np.array(test_acc_minmax), label='min-max scaled', color='red')
    plt.xlabel('epoch')
    plt.ylabel('Test R²')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('./figures/normalization.png')

