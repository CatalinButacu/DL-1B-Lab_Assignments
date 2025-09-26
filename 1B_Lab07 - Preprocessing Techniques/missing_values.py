import os
import csv 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class Regressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1)
                                 )
    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, test_loader, learn_rate = 0.01, num_epochs = 100):

        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learn_rate)

        train_losses = []
        test_accuracies = []

        for ep in range(num_epochs):
            self.train() 
            epoch_losses = []
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                yhat = self.forward(x_batch)
                loss = loss_func(yhat, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_mean_loss = sum(epoch_losses)/len(epoch_losses)
            train_losses.append(epoch_mean_loss)

            self.eval()
            with torch.no_grad():
                all_y_test = []
                all_yhat_test = []
                for x_test_batch, y_test_batch in test_loader:
                    yhat_test = self.forward(x_test_batch)
                    all_y_test.append(y_test_batch)
                    all_yhat_test.append(yhat_test)
                
                all_y_test = torch.cat(all_y_test)
                all_yhat_test = torch.cat(all_yhat_test)

                ss_res = torch.sum((all_y_test - all_yhat_test) ** 2)
                ss_tot = torch.sum((all_y_test - torch.mean(all_y_test)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                test_accuracies.append(r_squared.item())

            # print(f'Epoch: {ep}, Loss: {epoch_mean_loss:.4f}, Test R²: {r_squared.item():.4f}')

        return train_losses, test_accuracies

def get_data():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32).unsqueeze(1)
    return X, y
 
def simulate_missing_values(X, percentage = 0.1):
    rng = np.random.default_rng(seed = 42)
    mask = rng.uniform(0, 1, X.shape) < percentage
    X_missing = X.clone()
    X_missing[mask] = torch.nan
    return X_missing

def save_to_csv(results_for_percentage, percentage, output_dir):
    filename = f'results_{int(percentage*100)}_percent.csv'
    filepath = os.path.join(output_dir, filename)
    num_epochs = len(next(iter(results_for_percentage['loss'].values()))) 

    headers = ['Epoch']
    methods = ['mean', 'median', 'knn']
    for method in methods:
        headers.extend([f'{method.capitalize()}_Loss', f'{method.capitalize()}_Accuracy'])

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for epoch in range(num_epochs):
            row = [epoch]
            for method in methods:
                loss = results_for_percentage['loss'].get(method, [None]*num_epochs)[epoch]
                accuracy = results_for_percentage['accuracy'].get(method, [None]*num_epochs)[epoch]
                row.extend([loss if loss is not None else '', accuracy if accuracy is not None else ''])
            writer.writerow(row)
    print(f"Combined results saved to {filepath}")


if __name__ == '__main__':
    
    X_orig, y_orig = get_data()
    input_size = X_orig.shape[1]

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
    
    missing_percentages = [0.05, 0.10, 0.15]
    results = {}

    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    output_dir = './output/ex2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for percentage in missing_percentages:
        print(f"\n--- Processing {int(percentage*100)}% missing values ---")
        
        X_train_missing = simulate_missing_values(X_train_orig, percentage=percentage)
        X_test_complete = X_test_orig.clone()
        results[percentage] = {'loss': {}, 'accuracy': {}}

        # Mean imputation
        imputer_mean = SimpleImputer(strategy='mean')
        X_train_mean = torch.tensor(imputer_mean.fit_transform(X_train_missing.numpy()), dtype=torch.float32)
        X_test_mean = torch.tensor(imputer_mean.transform(X_test_complete.numpy()), dtype=torch.float32) # Use transform for test
        
        dataset_train_mean = TensorDataset(X_train_mean, y_train)
        dataset_test_mean = TensorDataset(X_test_mean, y_test) # Create test dataset
        loader_train_mean = DataLoader(dataset_train_mean, batch_size=64, shuffle=True)
        loader_test_mean = DataLoader(dataset_test_mean, batch_size=64, shuffle=False) # No shuffle for test

        print("Training with mean imputation...")
        reg_mean = Regressor(input_size)        
        losses, accuracies = reg_mean.train_model(loader_train_mean, loader_test_mean)
        results[percentage]['loss']['mean'] = losses
        results[percentage]['accuracy']['mean'] = accuracies


        # Median imputation        
        imputer_median = SimpleImputer(strategy='median')
        X_train_median = torch.tensor(imputer_median.fit_transform(X_train_missing.numpy()), dtype=torch.float32)
        X_test_median = torch.tensor(imputer_median.transform(X_test_complete.numpy()), dtype=torch.float32)

        dataset_train_median = TensorDataset(X_train_median, y_train)
        dataset_test_median = TensorDataset(X_test_median, y_test)
        loader_train_median = DataLoader(dataset_train_median, batch_size=64, shuffle=True)
        loader_test_median = DataLoader(dataset_test_median, batch_size=64, shuffle=False)

        print("Training with median imputation...")
        reg_median = Regressor(input_size)
        losses, accuracies = reg_median.train_model(loader_train_median, loader_test_median)
        results[percentage]['loss']['median'] = losses
        results[percentage]['accuracy']['median'] = accuracies


        # KNN imputation
        imputer_knn = KNNImputer(n_neighbors=5)
        X_train_knn = torch.tensor(imputer_knn.fit_transform(X_train_missing.numpy()), dtype=torch.float32)
        X_test_knn = torch.tensor(imputer_knn.transform(X_test_complete.numpy()), dtype=torch.float32)

        dataset_train_knn = TensorDataset(X_train_knn, y_train)
        dataset_test_knn = TensorDataset(X_test_knn, y_test)
        loader_train_knn = DataLoader(dataset_train_knn, batch_size=64, shuffle=True)
        loader_test_knn = DataLoader(dataset_test_knn, batch_size=64, shuffle=False)

        print("Training with KNN imputation...")
        reg_knn = Regressor(input_size)
        losses, accuracies = reg_knn.train_model(loader_train_knn, loader_test_knn)
        results[percentage]['loss']['knn'] = losses
        results[percentage]['accuracy']['knn'] = accuracies

        # Save combined results for this percentage
        save_to_csv(results[percentage], percentage, output_dir)

        # Plotting
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Convergence & Accuracy Comparison ({int(percentage*100)}% Missing Values)', fontsize=16)

        colors = {'mean': 'blue', 'median': 'orange', 'knn': 'green'}
        labels = {'mean': 'Mean', 'median': 'Median', 'knn': 'KNN'}
        any_plot_generated = False

        # Row 1: Loss Curves
        for i, method in enumerate(['mean', 'median', 'knn']):
            ax = axs[0, i]
            if results[percentage]['loss'].get(method):
                ax.plot(results[percentage]['loss'][method], label=f'{labels[method]} Loss', color=colors[method])
                ax.set_title(f'{labels[method]} Imputation - Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MSE Loss (Train)')
                ax.legend()
                ax.grid(True)
                any_plot_generated = True
            else:
                ax.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{labels[method]} Imputation - Loss')

        # Row 2: Accuracy (R-squared) Curves
        for i, method in enumerate(['mean', 'median', 'knn']):
            ax = axs[1, i]
            if results[percentage]['accuracy'].get(method):
                ax.plot(results[percentage]['accuracy'][method], label=f'{labels[method]} R-squared', color=colors[method])
                ax.set_title(f'{labels[method]} Imputation - Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('R-squared (Test)')
                ax.legend()
                ax.grid(True)
                ax.set_ylim(bottom=min(0, ax.get_ylim()[0])) # Adjust ylim per subplot
                any_plot_generated = True
            else:
                ax.text(0.5, 0.5, 'No Accuracy Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{labels[method]} Imputation - Accuracy')

        # Save the combined plot
        if any_plot_generated:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
            save_path = f'./figures/comparison_{int(percentage*100)}_percent_detailed.png' # New filename
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.close(fig)
        else:
            print(f"No results to plot for {int(percentage*100)}% missing values.")
            plt.close(fig)

    
    
