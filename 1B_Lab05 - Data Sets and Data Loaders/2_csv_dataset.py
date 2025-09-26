import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class IrisDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values

        if isinstance(self.labels[0], str):
            self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
            self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

if __name__ == '__main__':

    # load csv file
    file_path = 'data/iris.csv'
    ds = IrisDataset(file_path)
    data_loader = DataLoader(ds,shuffle = True)

    # fetch a single batch and print its shape
    features_batch, labels_batch = next(iter(data_loader))
    print(features_batch.shape) # torch.Size([1, 5])  // 1 sample with 5 features (Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    print(labels_batch.shape)   # torch.Size([1])    // 1 label (the flower Species)



