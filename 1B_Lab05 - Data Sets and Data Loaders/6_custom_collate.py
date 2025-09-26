import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# define a custom collate function to handle variable_size images
def custom_collate_func(batch):
    images, labels = zip(*batch)

    # find max height and width from batch
    max_height = max([image.shape[1] for image in images])
    max_width = max([image.shape[2] for image in images])

    # pad images to the max size in the batch
    # a useful function is torch.nn..functional.pad()
    padded_images = []
    for image in images:
        pad_height = max_height - image.shape[1]
        pad_width = max_width - image.shape[2]
        padded_image = torch.nn.functional.pad(image, (0, pad_width, 0, pad_height))
        padded_images.append(padded_image)

    images_tensor = torch.stack(padded_images)
    labels_tensor = torch.tensor(labels)

    return images_tensor, labels_tensor

# define transform that randomly resizes images in dataset
transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale = (0.5, 1.0)),
    transforms.ToTensor(),
    ])

dataset = datasets.CIFAR10(root='./data', train = True, transform = transform, download = True)

data_loader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = custom_collate_func)

images, labels = next(iter(data_loader))

print(images.shape) # torch.Size([8, 3, 32, 32])  // 8 images, 3 channels, 32 height, 32 width
print(labels.shape) # torch.Size([8])  // 8 labels for each image in the batch

