import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

image_transform = transforms.Compose([
    # transformations applied to images: resizing, normalization, conversion to tensor
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

# load image data set
fashion_mnist_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=image_transform)

# text reviews for each FashionMNIST label (0-9)
fashion_mnist_reviews = {
    0: "A very comfortable t-shirt, soft and breathable.",
    1: "Durable and stylish trousers, perfect for casual wear.",
    2: "A lovely pullover, keeps you warm in winter.",
    3: "A sleek dress, great for formal occasions.",
    4: "A simple and elegant coat, perfect for cold weather.",
    5: "Trendy sandals that are lightweight and comfortable.",
    6: "A stylish and functional shirt, fits well.",
    7: "Well-built sneakers, great for running.",
    8: "A durable and practical bag, spacious and strong.",
    9: "A classy ankle boot, suitable for every occasion."
}

# generate dataset entries where each FahionMNIST image has a corresponding review
# each entry is a tuple consising in (image, review, label) 
dataset_entries = []
for image, label in fashion_mnist_dataset:
    review = fashion_mnist_reviews[label]
    dataset_entries.append((image, review, label))

# build vocabulary for text tokenization
def build_vocab(data, min_freq = 1):
    counter = Counter()
    for _, text, _ in data:
        counter.update(word_tokenize(text.lower()))

    # vocabulary is a dictionary from word to numeric index
    vocab = {word: idx + 1 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab['<unk>'] = 0  # add a special token for unknown words

    return vocab

vocab = build_vocab(dataset_entries)

class MultiModalDataset(Dataset):
    def __init__(self, data_entries, vocab, transform):
        self.data_entries = data_entries
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        image, text, label = self.data_entries[idx]

        # convert image to PIL image and apply transformations
        image = transforms.ToPILImage()(image)
        image = self.transform(image)

        # tokenize and convert text to indices
        tokenized_text = word_tokenize(text.lower())
        indexed_text = [self.vocab.get(word, self.vocab['<unk>']) for word in tokenized_text] #indices from vocabulary
        
        return image, torch.tensor(indexed_text, dtype = torch.long), torch.tensor(label, dtype = torch.long)

# define a collate function for multimodal batching
def collate_func(batch):
    images, texts, labels = zip(*batch)

    # find max length in batch and padd all texts
    padded_texts = []
    max_len = max([len(text) for text in texts])
    for text in texts:
        pad_len = max_len - len(text)
        padded_text = torch.nn.functional.pad(text, (0, pad_len))
        padded_texts.append(padded_text)

    return torch.stack(images), torch.stack(padded_texts), torch.tensor(labels)

dataset = MultiModalDataset(dataset_entries, vocab, image_transform)
data_loader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = collate_func)

images_batch, texts_batch, labels_batch = next(iter(data_loader))

print(images_batch.shape) # torch.Size([8, 1, 28, 28])  // 8 images, 1 channel, 28 height, 28 width
print(texts_batch.shape)  # torch.Size([8, 11])  // 8 texts, 11 words 
print(labels_batch.shape) # torch.Size([8])  // 8 labels, one for each image in the batch




