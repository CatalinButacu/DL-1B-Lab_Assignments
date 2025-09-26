import os
import numpy as np
import polars as pl
import csv

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from matplotlib import pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

def load_and_preprocess_data(file_path='data/bank-additional.csv'):
    """Loads the bank dataset, preprocesses features using sklearn, and returns X, y, and feature names."""
    try:
        data = pl.read_csv(file_path, separator=';')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None, None

    target_col = 'y'
    y = data[target_col].to_numpy()
    y = np.where(y == 'yes', 1, 0).astype(np.int8)

    X_df = data.drop(target_col)

    categorical_cols = [col for col in X_df.columns if X_df[col].dtype == pl.Utf8]
    numerical_cols = [col for col in X_df.columns if X_df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]]

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X_df)

    num_feature_names = numerical_cols
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    final_feature_cols = num_feature_names + list(cat_feature_names)

    return X_processed.astype(np.float32), y, final_feature_cols


class Classifier(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def train_classifier(model, criterion, optimizer, train_loader, X_test, y_test, num_epochs=50, patience=5):
    """Trains the classifier model with early stopping and returns training metrics."""
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    best_test_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            predicted = (test_outputs > 0.5).float()
            accuracy = (predicted == y_test).float().mean().item()

            history['test_loss'].append(test_loss)
            history['test_accuracy'].append(accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                break

    metrics = [
        {'epoch': i, 'train_loss': tl, 'test_loss': vl, 'test_accuracy': ta}
        for i, (tl, vl, ta) in enumerate(zip(history['train_loss'], history['test_loss'], history['test_accuracy']))
    ]

    return history['train_loss'], history['test_loss'], history['test_accuracy'], metrics


if __name__ == '__main__':
    os.makedirs('output/ex3', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    X, y, feature_names = load_and_preprocess_data()

    if X is None or y is None:
        print("Failed to load or process data. Exiting.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = Classifier(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses, test_accuracies, metrics = train_classifier(
        model, criterion, optimizer, train_loader, X_test_tensor, y_test_tensor
    )
    
    with open('output/ex3/category_encoding_bank_dataset.csv', 'w', newline='') as csvfile:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCELoss)')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig('figures/category_encoding_bank_dataset.png')

