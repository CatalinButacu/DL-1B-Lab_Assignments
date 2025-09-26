import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')

file_path = './data/sentiment.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Process dataset into DataFrame
data = [line.strip().split('\t', 1) for line in lines]  # Ensure splitting properly
df = pd.DataFrame(data, columns=['label', 'text'])
df['label'] = df['label'].astype(int)  # Convert label to integer

# Tokenization function
def tokenize(text):
    return word_tokenize(text.lower())

# Build vocabulary
def build_vocab(df, min_freq=1):
    counter = Counter()
    for text in df['text']:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {'<PAD>': 0, '<UNK>': 1}  # Add special tokens
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab

vocab = build_vocab(df)

# Custom dataset
class SentimentDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.data = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        tokenized_text = tokenize(text)
        indexed_text = [self.vocab.get(word, self.vocab['<UNK>']) for word in tokenized_text]

        return torch.tensor(indexed_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Collate function for padding sequences
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)

    return padded_texts, torch.tensor(labels, dtype=torch.long)

# DataLoader
dataset = SentimentDataset(df, vocab)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Testing batch output
text_batch, label_batch = next(iter(data_loader))
print(text_batch) # Padded sequence of tokens/words (indexed)
print(label_batch) # Labels (positive/negative)
