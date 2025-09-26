import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

import os
os.makedirs('results', exist_ok=True)

random_seed = 42
torch.manual_seed(random_seed)

### SELF MADE PERCEPTRON CLASS ###

class Perceptron:
    def __init__(self, input_size): # output size = 1
        self.input_size = input_size
        self.w = torch.zeros([input_size, 1]) # w is a column vector
        self.b = 0.0
        self.loss_history = list()
        self.acc_history = list()

    def init_weights(self, min_val, max_val):
        # initialize weights with values between min_val and max_val
        self.w = torch.FloatTensor(self.input_size, 1).uniform_(min_val, max_val)
        self.b = torch.FloatTensor(1).uniform_(min_val, max_val)

    def sigmoid_activation(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, inputs):
        return self.sigmoid_activation(torch.mm(inputs, self.w) + self.b)

    def compute_loss(self, yhat, y):
        return 0.5 * torch.mean((yhat-y)**2)

    def train(self, X_train, y_train, learn_rate, num_epochs):

        self.init_weights(-1.0, 1.0)
        y = y_train.view(-1, 1).float() # y is a column vector

        for _ in range(num_epochs):

            # prediction
            yhat = self.forward(X_train)

            # metrics computing
            loss = self.compute_loss(yhat, y)
            acc = self.evaluate(X_train, y_train)
            self.loss_history.append(loss.item())
            self.acc_history.append(acc)


            # goal: compute gradients of loss function L w.r.t. weights w and biases b
            # w.r.t = "with regard to"            
        
            # gradient of L w.r.t activation yhat
            dL_dyhat = yhat - y

            # gradient of the activation function w.r.t. output z
            dyhat_dz = yhat * (1 - yhat)

            # gradient of the loss w.r.t. output z using chain rule
            dL_dz = dL_dyhat * dyhat_dz


            # gradient of the perceptron output w.r.t weights
            # apply chain rule to determine gradient of loss w.r.t weights
            dL_dw = X_train.T @ dL_dz / X_train.size(0)

            # for the biases, dz/db = 1:
            dL_db = torch.mean(dL_dz)

            # update weights using gradient descent
            self.w = self.w - learn_rate * dL_dw
            self.b = self.b - learn_rate * dL_db

    def evaluate(self, X_test, y_test):
        yhat = self.forward(X_test)
        yhat = (yhat > 0.5).float()
        acc = (yhat == y_test.view(-1, 1)).float().mean()
        return acc.item()

def generate_dataset(num_instances, num_features, random_seed = None):
    X, y = make_classification(n_samples = num_instances,
                               n_features = num_features,
                               n_informative = num_features, 
                               n_redundant = 0,
                               n_clusters_per_class = 1,
                               class_sep = 0.5,
                               random_state = random_seed)
    return torch.tensor(X).float(), torch.tensor(y).long()

### PYTORCH PERCEPTRON CLASS ###

class PytorchPerceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.loss_history = list()
        self.acc_history = list()

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def train(self, X, y, learn_rate, num_epochs):
        optimizer = optim.SGD(self.parameters(), lr=learn_rate)
        criterion = nn.MSELoss()
        
        for _ in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(X).squeeze()
            
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()
            
            # Track metrics
            self.loss_history.append(loss.item())
            with torch.no_grad():
                predicted = (outputs > 0.5).float()
                acc = (predicted == y).float().mean().item()
                self.acc_history.append(acc)

def plot_progress(num_epochs, scratch_loss, pytorch_loss, scratch_acc, pytorch_acc):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), scratch_loss, label='Scratch Perceptron')
    plt.plot(range(num_epochs), pytorch_loss, label='PyTorch Perceptron')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), scratch_acc, label='Scratch Perceptron')
    plt.plot(range(num_epochs), pytorch_acc, label='PyTorch Perceptron')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/perceptron_comparison.png')
    plt.close()

if __name__ == '__main__':
    # Data
    num_instances = 1000
    num_features = 5
    num_epochs = 2000
    lr = 0.1
    
    # Generate dataset
    X, y = generate_dataset(num_instances, num_features, random_seed)
    
    # Train own perceptron
    scratch_perc = Perceptron(num_features)
    scratch_perc.train(X, y, lr, num_epochs)
    
    # Train PyTorch perceptron
    torch_perc = PytorchPerceptron(num_features)
    torch_perc.train(X, y, lr, num_epochs)
    
    # Generate comparison plots
    plot_progress(num_epochs, 
                 scratch_perc.loss_history, torch_perc.loss_history,
                 scratch_perc.acc_history, torch_perc.acc_history)
    
    # Final evaluation
    print(f"Scratch Perceptron - Final Loss: {scratch_perc.loss_history[-1]:.4f}")
    print(f"PyTorch Perceptron - Final Loss: {torch_perc.loss_history[-1]:.4f}")

    print(f"Scratch Perceptron - Final Accuracy: {scratch_perc.evaluate(X, y):.4f}")
    print(f"PyTorch Perceptron - Final Accuracy: {torch_perc.acc_history[-1]:.4f}")      

