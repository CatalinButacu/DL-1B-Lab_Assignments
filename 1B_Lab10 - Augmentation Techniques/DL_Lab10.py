import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16

            nn.Conv2d(32, 64, kernel_size = 3, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8

            nn.Conv2d(64, 128, kernel_size = 3, padding = 'same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # 1x1
            )
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    def mixup_data(self, x, y, alpha=1.0):
        '''Performs MixUp on the input data and labels'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y, alpha=1.0):
        '''Performs CutMix on the input data and labels'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def rand_bbox(self, size, lam):
        '''Generate random bounding box for CutMix'''
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def train_model(self, train_loader, test_loader, num_epochs = 10, learn_rate = 0.001, mixup_alpha=1.0, cutmix_alpha=1.0, mixup_prob=1.0):
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr = learn_rate)
        self.train()
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        for ep in range(num_epochs):
            ep_loss = 0.0
            correct = 0
            total = 0
            
            # Training phase
            for images, labels in train_loader:
                # Randomly choose between MixUp, CutMix, or no augmentation
                r = np.random.rand(1)
                if r < mixup_prob:
                    if r < mixup_prob/2:  # 50-50 chance for each
                        # MixUp
                        mixed_images, labels_a, labels_b, lam = self.mixup_data(images, labels, mixup_alpha)
                    else:  
                        # CutMix
                        mixed_images, labels_a, labels_b, lam = self.cutmix_data(images, labels, cutmix_alpha)
                    
                    outputs = self.forward(mixed_images)
                    loss = lam * loss_func(outputs, labels_a) + (1 - lam) * loss_func(outputs, labels_b)
                else:
                    outputs = self.forward(images)
                    loss = loss_func(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                ep_loss += loss.item()
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                if r < mixup_prob:
                    correct += (lam * (predicted == labels_a).float() + (1 - lam) * (predicted == labels_b).float()).sum().item()
                else:
                    correct += (predicted == labels).sum().item()
            
            train_loss = ep_loss / len(train_loader)
            train_acc = correct / total
            
            # Validation phase
            val_acc = self.eval_model(test_loader, verbose=False)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {ep+1}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}')
        
        self.plot_training_history(history)

    def eval_model(self, test_loader, verbose=True):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.forward(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct/total
        if verbose:
            print(f'Acc: {accuracy:.3f}')
        return accuracy

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))

        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot 2: Training and Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./plots/2.3_training_history_both.png')
        plt.close()

if __name__ == '__main__':

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, 
                                                 transform = transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, 
                                                transform = transform_test)

    # use a subset of CIFAR10 for faster training
    train_size = 5000
    test_size = 1000

    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)

    train_indices, _ = train_test_split(np.arange(len(train_dataset)), 
                                        train_size = train_size, 
                                        stratify = train_targets, 
                                        random_state = random_seed)
    
    test_indices, _ = train_test_split(np.arange(len(test_dataset)),
                                       train_size = test_size,
                                       stratify = test_targets,
                                       random_state = random_seed)

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

    model = CNN()
    model.train_model(train_loader, test_loader, num_epochs = 10, learn_rate = 0.001)
    
    model.eval_model(test_loader)


       










