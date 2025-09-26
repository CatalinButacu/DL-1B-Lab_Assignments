import torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
fashion_mnist_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# label ambiguity: modify labels from fashion mnist so that they don't coincide with the labels from mnist
for i in range(len(fashion_mnist_dataset)):
    img, label = fashion_mnist_dataset[i]
    fashion_mnist_dataset.targets[i] = label + 10

# merge datasets
merged_dataset = ConcatDataset([mnist_dataset, fashion_mnist_dataset])

# create dataloader for merged dataset
merged_loader = DataLoader(merged_dataset, batch_size = 8, shuffle = True)

# retrieve a batch of images
images, labels = next(iter(merged_loader))
# print shape of images and labels
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# display images
fig, axes = plt.subplots(1, 8, figsize = (15, 2))
for i in range(8):
    ax = axes[i]
    ax.imshow(images[i].squeeze(), cmap = 'gray')
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis('off')
plt.show()
