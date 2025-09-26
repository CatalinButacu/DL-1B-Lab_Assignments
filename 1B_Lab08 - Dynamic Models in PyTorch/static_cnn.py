import torch
import torch.nn as nn

class StaticCNN(nn.Module):
    def __init__(self, num_classes = 3, image_size = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding = 'same')
        self.bn1 = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(2) 

        self.conv2 = nn.Conv2d(16, 32, 3, padding = 'same')
        self.bn2 = nn.BatchNorm2d(32) 
        self.pool2 = nn.MaxPool2d(2) 

        self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same')
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        
        flat_features = 64 * (image_size // 4) * (image_size // 4) 
        self.fc = nn.Linear(flat_features, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

    def train_model(self, loader, num_epochs = 20, learn_rate = 0.001):
        self.train()
        lossFunc = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learn_rate)
        history = {'loss': [], 'accuracy': []}

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            correct_train = 0
            total_train = 0
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = lossFunc(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            avg_loss = total_loss / len(loader)
            epoch_train_accuracy = 100 * correct_train / total_train

            history['loss'].append(avg_loss)
            history['accuracy'].append(epoch_train_accuracy)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Ep {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {epoch_train_accuracy:.2f}%')
        
        return history

    def evaluate_model(self, loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy