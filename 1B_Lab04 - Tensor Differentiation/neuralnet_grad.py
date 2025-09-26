import torch

torch.manual_seed(42)

class NNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NNet, self).__init__()

        hidden_size1 = 20
        hidden_size2 = 10
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, output_size)
        self.relu = torch.nn.ReLU()
        self.debug = False
        self.loss_history = list()
        self.accuracy_history = list()


    def forward(self, X):
        y_pred = self.relu(self.fc1(X))
        y_pred = self.relu(self.fc2(y_pred))
        return self.fc3(y_pred)

    def train_model(self, X_train, y_train, learn_rate, num_epochs, lossFunc = torch.nn.CrossEntropyLoss()):
        optimizer = torch.optim.SGD(self.parameters(), lr=learn_rate)        
        self.loss_history.clear()
        self.accuracy_history.clear()

        for ep in range(num_epochs):
            optimizer.zero_grad()  # Reset gradients

            yhat = self.forward(X_train)  # Forward pass

            if isinstance(lossFunc, type) and issubclass(lossFunc, torch.autograd.Function):
                loss = lossFunc.apply(yhat, y_train)
            elif isinstance(lossFunc, torch.nn.MSELoss):
                y_train_one_hot = torch.zeros_like(yhat)
                y_train_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
                loss = lossFunc(yhat, y_train_one_hot)
            else:
                loss = lossFunc(yhat, y_train)
            
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            current_accuracy = self.evaluate(X_train, y_train)
            self.accuracy_history.append(current_accuracy)
            self.loss_history.append(loss.item())

            if ep % 10 == 0 and self.debug:
                print(f"Epoch {ep}: Loss = {loss.item():.4f}")


    def train_model_explicit(self, X_train, y_train, learn_rate, num_epochs, lossFunc = torch.nn.CrossEntropyLoss()):
        # train model while explicitly computing gradients        
        self.loss_history.clear()
        self.accuracy_history.clear()
        for ep in range(num_epochs):

            yhat = self.forward(X_train)

            if isinstance(lossFunc, type) and issubclass(lossFunc, torch.autograd.Function):
                loss = lossFunc.apply(yhat, y_train)
            elif isinstance(lossFunc, torch.nn.MSELoss):
                y_train_one_hot = torch.zeros_like(yhat)
                y_train_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
                loss = lossFunc(yhat, y_train_one_hot)
            else:
                loss = lossFunc(yhat, y_train)

            # gradients of loss w.r.t. weights and biases of all layers
            grad_w1 = torch.autograd.grad(loss, self.fc1.weight, retain_graph=True)[0]
            grad_b1 = torch.autograd.grad(loss, self.fc1.bias, retain_graph=True)[0]
            grad_w2 = torch.autograd.grad(loss, self.fc2.weight, retain_graph=True)[0]
            grad_b2 = torch.autograd.grad(loss, self.fc2.bias, retain_graph=True)[0]
            grad_w3 = torch.autograd.grad(loss, self.fc3.weight, retain_graph=True)[0]
            grad_b3 = torch.autograd.grad(loss, self.fc3.bias, retain_graph=True)[0]

            with torch.no_grad():
                # update weights and biases using gradient descent
                self.fc1.weight -= learn_rate * grad_w1
                self.fc1.bias -= learn_rate * grad_b1
                self.fc2.weight -= learn_rate * grad_w2
                self.fc2.bias -= learn_rate * grad_b2
                self.fc3.weight -= learn_rate * grad_w3
                self.fc3.bias -= learn_rate * grad_b3

            current_accuracy = self.evaluate(X_train, y_train)
            self.accuracy_history.append(current_accuracy)
            self.loss_history.append(loss.item())
            
            if ep % 10 == 0 and self.debug:
                print(f"Epoch {ep}: Loss = {loss.item():.4f}")
            
                
    def train_model_momentum(self, X_train, y_train, learn_rate, num_epochs, beta=0.9, lossFunc = torch.nn.CrossEntropyLoss()):
        # train model with momentum
        self.loss_history.clear()
        self.accuracy_history.clear()

        # initialize velocity terms for weights and biases
        v_w1 = torch.zeros_like(self.fc1.weight)
        v_b1 = torch.zeros_like(self.fc1.bias)
        v_w2 = torch.zeros_like(self.fc2.weight)
        v_b2 = torch.zeros_like(self.fc2.bias)
        v_w3 = torch.zeros_like(self.fc3.weight)
        v_b3 = torch.zeros_like(self.fc3.bias)

        for ep in range(num_epochs):

            yhat = self.forward(X_train)

            if isinstance(lossFunc, type) and issubclass(lossFunc, torch.autograd.Function):
                loss = lossFunc.apply(yhat, y_train)
            elif isinstance(lossFunc, torch.nn.MSELoss):
                y_train_one_hot = torch.zeros_like(yhat)
                y_train_one_hot.scatter_(1, y_train.unsqueeze(1), value=1.75)
                loss = lossFunc(yhat, y_train_one_hot)
            else:
                loss = lossFunc(yhat, y_train)

            # gradients of loss w.r.t. weights and biases of all layers
            grad_w1 = torch.autograd.grad(loss, self.fc1.weight, retain_graph=True)[0]
            grad_b1 = torch.autograd.grad(loss, self.fc1.bias, retain_graph=True)[0]
            grad_w2 = torch.autograd.grad(loss, self.fc2.weight, retain_graph=True)[0]
            grad_b2 = torch.autograd.grad(loss, self.fc2.bias, retain_graph=True)[0]
            grad_w3 = torch.autograd.grad(loss, self.fc3.weight, retain_graph=True)[0]
            grad_b3 = torch.autograd.grad(loss, self.fc3.bias, retain_graph=True)[0]

            with torch.no_grad():
                # update velocity terms
                v_w1 = beta * v_w1 - learn_rate * grad_w1
                v_b1 = beta * v_b1 - learn_rate * grad_b1
                v_w2 = beta * v_w2 - learn_rate * grad_w2
                v_b2 = beta * v_b2 - learn_rate * grad_b2
                v_w3 = beta * v_w3 - learn_rate * grad_w3
                v_b3 = beta * v_b3 - learn_rate * grad_b3

                # update weights and biases using momentum
                self.fc1.weight += v_w1
                self.fc1.bias += v_b1
                self.fc2.weight += v_w2
                self.fc2.bias += v_b2
                self.fc3.weight += v_w3
                self.fc3.bias += v_b3

            current_accuracy = self.evaluate(X_train, y_train)
            self.accuracy_history.append(current_accuracy)
            self.loss_history.append(loss.item())

            if ep % 10 == 0 and self.debug:
                print(f"Epoch {ep}: Loss = {loss.item():.4f}")
    
    def evaluate(self, X_test, y_test):
        # get accuracy of model
        yhat = self.forward(X_test)
        _, predicted = torch.max(yhat, 1)
        correct = (predicted == y_test).sum().item()
        return correct / len(y_test)

def generate_dataset(num_instances, num_features, num_classes, noise=0.5):
    features = []
    labels = []

    instances_per_class = num_instances // num_classes
    instances_per_class = [instances_per_class] * num_classes
    instances_per_class[0] += num_instances - sum(instances_per_class)

    for i in range(num_classes):
        X = torch.randn(instances_per_class[i], num_features) * noise + torch.full([instances_per_class[i], num_features], i)
        y = torch.full([instances_per_class[i]], i)

        features.append(X)
        labels.append(y)

    X = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)

    indices = torch.randperm(num_instances)
    X, y = X[indices], y[indices]

    return X, y

if __name__ == '__main__':
    
    num_instances = 100
    num_features = 10
    num_classes = 6
    data_noise = 0.5
    X, y = generate_dataset(num_instances, num_features, num_classes, data_noise)

    trainSplit = 0.8
    num_train_instances = int(num_instances * trainSplit)

    X_train, y_train = X[:num_train_instances], y[:num_train_instances]
    X_test, y_test = X[num_train_instances:], y[num_train_instances:]

    model = NNet(num_features, num_classes)
    
    learn_rate = 0.1
    num_epochs = 200

    model.train_model(X_train, y_train, learn_rate, num_epochs)

    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)

    print(f'Train accuracy: {train_acc:.4f}')
    print(f'Test accuracy: {test_acc:.4f}')
