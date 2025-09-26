import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets, transforms


def load_iris_dataset(test_split=0.2, random_seed=42):
    """
    Load Iris dataset as PyTorch tensors
    """
    # For simplicity, we're creating a synthetic iris-like dataset
    np.random.seed(random_seed)
    
    # 150 samples, 4 features (simulating Iris)
    num_samples = 150
    features = np.random.randn(num_samples, 4) * 0.5
    
    # 3 classes (0, 1, 2)
    labels = np.zeros(num_samples, dtype=np.int64)
    labels[50:100] = 1
    labels[100:] = 2
    
    # Add some class separation to make classes more distinguishable
    for i in range(num_samples):
        class_idx = labels[i]
        features[i, class_idx] += 2.0  # Make one feature stronger for each class
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    # Split into train and test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    return train_dataset, test_dataset


def load_fashion_mnist_dataset(test_split=0.2, random_seed=42):
    """
    Load Fashion MNIST dataset
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load the data
    try:
        # Try to load from the default directory, or download if not found
        train_data = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        test_data = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        return train_data, test_data
    
    except Exception as e:
        print(f"Error loading Fashion MNIST: {e}")
        # Return empty datasets
        return TensorDataset(torch.Tensor([]), torch.LongTensor([])), TensorDataset(torch.Tensor([]), torch.LongTensor([]))


def load_custom_dataset(num_instances=1000, num_features=8, num_classes=4, 
                        class_separation=1.0, noise=0.5, test_split=0.2, random_seed=42):
    """
    Generate a custom dataset with specified parameters
    """
    np.random.seed(random_seed)
    
    # Generate synthetic data
    features = np.random.randn(num_instances, num_features) * noise
    labels = np.random.randint(0, num_classes, size=num_instances)
    
    # Add separation between classes
    for i in range(num_instances):
        class_idx = labels[i]
        # Add a bias to certain features based on class
        for j in range(min(num_classes, num_features)):
            if j == class_idx:
                features[i, j] += class_separation
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    # Split into train and test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    return train_dataset, test_dataset