import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# define transformations
transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ## 3 channels, normalized to [-1, 1]
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10)
    ])

# load cifar dataset with transforms
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# get a batch of images
images, labels = next(iter(data_loader))

# display transformed images
fig, axes = plt.subplots(1, 8, figsize = (15, 2))
for i in range(8):
    ax = axes[i]
    ax.imshow(images[i].permute(1, 2, 0))
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')
plt.show()





