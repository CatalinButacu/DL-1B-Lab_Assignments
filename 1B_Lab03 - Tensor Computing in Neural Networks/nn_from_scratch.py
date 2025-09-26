import torch
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import os
os.makedirs('results', exist_ok=True)

torch.manual_seed(42)

class NNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.w1 = torch.zeros(input_size, hidden_size) # weights inbetween the input and hidden layers
        self.b1 = torch.zeros(hidden_size) # bias inbetween the input and hidden layers
        self.z1 = torch.zeros(hidden_size) # outputs of hidden layer (inactivated)
        self.a1 = torch.zeros(hidden_size) # activations of hidden layer
        
        self.w2 = torch.zeros(hidden_size, output_size) # weights inbetween the hidden and output layers
        self.b2 = torch.zeros(output_size) # bias inbetween hidden and output layers
        self.z2 = torch.zeros(output_size) # outputs of output layer

        self.train_loss = list()
        self.train_acc = list()

        self.init_weights(-1.0, 1.0)

    def init_weights(self, min_weight, max_weight):
        # init weights and biases with random values from [min_weight, max_weight]
        self.w1 = torch.FloatTensor(self.w1.size()).uniform_(min_weight, max_weight)
        self.b1 = torch.FloatTensor(self.b1.size()).uniform_(min_weight, max_weight)
        self.w2 = torch.FloatTensor(self.w2.size()).uniform_(min_weight, max_weight)
        self.b2 = torch.FloatTensor(self.b2.size()).uniform_(min_weight, max_weight)
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def sigmoid_gradient(self, x):
        return x * (1 - x)

    def softmax(self, x):
        e = torch.exp(x)
        return e / torch.sum(e, dim = 1).unsqueeze(1)

    def cross_entropy_loss(self, y, yhat):
        log_probs = torch.log_softmax(yhat, dim=1)
        return -torch.mean(torch.sum(y * log_probs, dim=1))

    def forward(self, inputs):
        # forward propagation
        self.z1 = torch.matmul(inputs, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = torch.matmul(self.a1, self.w2) + self.b2
        return self.softmax(self.z2)

    def train(self, X_train, y_train, learn_rate, num_epochs):

        self.train_loss = list()
        self.train_acc = list()
        
        for _ in range(num_epochs):

            # prediction 
            yhat = self.forward(X_train)

            # metrics computing
            loss = self.cross_entropy_loss(y_train, yhat)
            acc = self.evaluate(X_train, y_train)
            self.train_loss.append(loss.item())
            self.train_acc.append(acc)

            # gradient of loss w.r.t. output
            batch_size = X_train.size(0)
            dL_dz2 = (yhat - y_train) / batch_size

            # gradient of loss w.r.t. output layer weights
            dL_dw2 = torch.matmul(self.a1.t(), dL_dz2)

            # gradient of loss w.r.t. output biases
            dL_db2 = torch.sum(dL_dz2, dim = 0)

            # gradient of loss w.r.t. hidden layer output
            dL_dz1 = torch.matmul(dL_dz2, self.w2.t()) * self.sigmoid_gradient(self.a1)

            # gradient of loss w.r.t. hidden layer weights
            dL_dw1 = torch.matmul(X_train.t(), dL_dz1)

            # gradient of loss w.r.t. hidden layer biases
            dL_db1 = torch.sum(dL_dz1, dim = 0)

            # update weights and biases using gradient descent
            self.w1 = self.w1 - learn_rate * dL_dw1
            self.b1 = self.b1 - learn_rate * dL_db1
            self.w2 = self.w2 - learn_rate * dL_dw2 
            self.b2 = self.b2 - learn_rate * dL_db2
            
    def evaluate(self, X_test, y_test):
        yhat = self.forward(X_test)
        yhat = torch.argmax(yhat, dim = 1)
        y = torch.argmax(y_test, dim = 1)
        acc = torch.sum(y == yhat).item() / yhat.size(0)
        return acc


class TorchNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.train_loss = list()
        self.train_acc = list()
        
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_model(self, X_train, y_train, learn_rate=0.1, num_epochs=100):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learn_rate)
        self.train_loss.clear()
        self.train_acc.clear()
        
        for _ in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(X_train)
            
            # Compute loss
            loss = criterion(outputs, torch.argmax(y_train, dim=1))
            loss.backward()
            optimizer.step()
            
            # Track metrics
            self.train_loss.append(loss.item())
            with torch.no_grad():
                acc = (torch.argmax(outputs, dim=1) == torch.argmax(y_train, dim=1)).float().mean()
                self.train_acc.append(acc.item())
    
    def evaluate(self, X_test, y_test):
        with torch.no_grad():
            outputs = self(X_test)
            y_pred = torch.argmax(outputs, dim=1)
            y_true = torch.argmax(y_test, dim=1)
            return (y_pred == y_true).float().mean().item()

def plot_progress(num_epochs, scratch_loss, pytorch_loss, scratch_acc, pytorch_acc):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), scratch_loss, label='Scratch NN')
    plt.plot(range(num_epochs), pytorch_loss, label='PyTorch NN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), scratch_acc, label='Scratch NN')
    plt.plot(range(num_epochs), pytorch_acc, label='PyTorch NN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/nn_comparison.png')
    plt.close()

if __name__ == '__main__':

    # load data set
    dataset = load_iris()
    X = torch.from_numpy(dataset.data).float()
    y = torch.from_numpy(dataset.target).long()
    # convert y to one-hot encodings
    y = torch.nn.functional.one_hot(y, num_classes = len(dataset.target_names)).float()

    # shuffle data
    rand_idx = torch.randperm(X.shape[0])
    X = X[rand_idx]
    y = y[rand_idx]

    num_epochs = 400
    lr = 0.1
    input_size, hidden_size, output_size = 4, 5, 3

    # scratch implementation
    nnet = NNet(input_size, hidden_size, output_size)    
    nnet.train(X, y, learn_rate = lr, num_epochs = num_epochs)
    acc = nnet.evaluate(X, y)
    print(f'Scratch accuracy: {acc}')

    # torch implementation
    model = TorchNN(input_size, hidden_size, output_size)
    model.train_model(X, y, learn_rate=lr, num_epochs=num_epochs)
    acc = model.evaluate(X, y)
    print(f'PyTorch accuracy: {acc}')

    # results
    plot_progress(num_epochs, 
                  nnet.train_loss, model.train_loss, 
                  nnet.train_acc, model.train_acc)




