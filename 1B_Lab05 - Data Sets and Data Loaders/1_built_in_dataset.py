import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))
    ])

# define MNIST train and test data sets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# define train and test loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# get a batch of data
data_iterator = iter(train_loader)
images, labels = next(data_iterator)

# display the first 10 images from the batch
fig, axes = plt.subplots(1, 10, figsize = (12, 2))
for i in range(10):
    axes[i].imshow(images[i].squeeze(), cmap = 'gray') # remove channel dimension
    axes[i].set_title(f'Label = {labels[i].item()}')
    axes[i].axis('off')

plt.show()

