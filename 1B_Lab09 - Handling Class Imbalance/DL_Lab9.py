import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

random_seed = 42
torch.manual_seed(42)

def gen_imbalanced_dataset(num_instances = 1000, num_features = 10, imbalance_ratio = 0.1, random_state = None):
    # imbalance ratio = proportion of minority class
    weights = [1 - imbalance_ratio, imbalance_ratio]
    X, y = make_classification(n_samples = num_instances, 
                               n_features = num_features, 
                               n_informative = num_features,
                               n_redundant = 0,
                               n_repeated = 0,
                               n_classes = 2,
                               class_sep = 0.7,
                               weights = weights, 
                               flip_y = 0.01,
                               random_state = random_state)
    return X, y

class NNet(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)        
        return x

    def train_model(self, train_loader, num_epochs = 10, learn_rate = 0.001, pos_weight = None):
        loss_func = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        optimizer = torch.optim.Adam(self.parameters(), lr = learn_rate)
        for ep in range(num_epochs):
            ep_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = loss_func(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            print(f'Epoch: {ep+1}, Loss: {ep_loss / len(train_loader):.4f}')
    
    def evaluate(self, test_loader):
        self.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.forward(inputs)
                predicted = torch.sigmoid(outputs) >= 0.5
                
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.numpy().flatten())
        
        # Convert to numpy arrays for easier calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)}
        }

       
if __name__ == '__main__':

    num_instances = 1000
    num_features = 10
    imbalance_ratio = 0.2
    X, y = gen_imbalanced_dataset(num_instances = num_instances, 
                                    num_features = num_features, 
                                    imbalance_ratio = imbalance_ratio, 
                                    random_state = random_seed)

    # number of instances for each label
    num_zero, num_one = np.bincount(y)
    print(f"Class distribution - Class 0: {num_zero}, Class 1: {num_one}, Ratio: {num_one/num_zero:.3f}")

    # stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        stratify = y, 
                                                        random_state = random_seed)

    # Create test dataset and loader (same for all experiments)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Dictionary to store results
    results = {}
    
    # 1. Baseline model (no handling of class imbalance)
    print("\n1. Training baseline model (no handling of class imbalance)")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    baseline_model = NNet(num_features)
    baseline_model.train_model(train_loader, num_epochs=10, learn_rate=0.001)
    results['baseline'] = baseline_model.evaluate(test_loader)
    print("Baseline model evaluation:")
    for metric, value in results['baseline'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 2. Class weighting
    print("\n2. Training with class weighting")
    # Calculate weight for positive class
    pos_weight = torch.tensor([num_zero / num_one])
    print(f"Positive class weight: {pos_weight.item():.4f}")
    
    weighted_model = NNet(num_features)
    weighted_model.train_model(train_loader, num_epochs=10, learn_rate=0.001, pos_weight=pos_weight)
    results['weighted'] = weighted_model.evaluate(test_loader)
    print("Weighted model evaluation:")
    for metric, value in results['weighted'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 3. Random Oversampling
    print("\n3. Training with Random Oversampling")
    ros = RandomOverSampler(random_state=random_seed)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    print(f"After Random Oversampling - Class 0: {sum(y_train_ros == 0)}, Class 1: {sum(y_train_ros == 1)}")
    
    X_train_ros_tensor = torch.tensor(X_train_ros, dtype=torch.float32)
    y_train_ros_tensor = torch.tensor(y_train_ros, dtype=torch.float32)
    train_dataset_ros = TensorDataset(X_train_ros_tensor, y_train_ros_tensor)
    train_loader_ros = DataLoader(train_dataset_ros, batch_size=64, shuffle=True)
    
    ros_model = NNet(num_features)
    ros_model.train_model(train_loader_ros, num_epochs=10, learn_rate=0.001)
    results['random_oversampling'] = ros_model.evaluate(test_loader)
    print("Random Oversampling model evaluation:")
    for metric, value in results['random_oversampling'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 4. SMOTE
    print("\n4. Training with SMOTE")
    smote = SMOTE(random_state=random_seed)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Class 0: {sum(y_train_smote == 0)}, Class 1: {sum(y_train_smote == 1)}")
    
    X_train_smote_tensor = torch.tensor(X_train_smote, dtype=torch.float32)
    y_train_smote_tensor = torch.tensor(y_train_smote, dtype=torch.float32)
    train_dataset_smote = TensorDataset(X_train_smote_tensor, y_train_smote_tensor)
    train_loader_smote = DataLoader(train_dataset_smote, batch_size=64, shuffle=True)
    
    smote_model = NNet(num_features)
    smote_model.train_model(train_loader_smote, num_epochs=10, learn_rate=0.001)
    results['smote'] = smote_model.evaluate(test_loader)
    print("SMOTE model evaluation:")
    for metric, value in results['smote'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 5. ADASYN
    print("\n5. Training with ADASYN")
    adasyn = ADASYN(random_state=random_seed)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    print(f"After ADASYN - Class 0: {sum(y_train_adasyn == 0)}, Class 1: {sum(y_train_adasyn == 1)}")
    
    X_train_adasyn_tensor = torch.tensor(X_train_adasyn, dtype=torch.float32)
    y_train_adasyn_tensor = torch.tensor(y_train_adasyn, dtype=torch.float32)
    train_dataset_adasyn = TensorDataset(X_train_adasyn_tensor, y_train_adasyn_tensor)
    train_loader_adasyn = DataLoader(train_dataset_adasyn, batch_size=64, shuffle=True)
    
    adasyn_model = NNet(num_features)
    adasyn_model.train_model(train_loader_adasyn, num_epochs=10, learn_rate=0.001)
    results['adasyn'] = adasyn_model.evaluate(test_loader)
    print("ADASYN model evaluation:")
    for metric, value in results['adasyn'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 6. Tomek Links
    print("\n6. Training with Tomek Links")
    tl = TomekLinks()
    X_train_tl, y_train_tl = tl.fit_resample(X_train, y_train)
    print(f"After Tomek Links - Class 0: {sum(y_train_tl == 0)}, Class 1: {sum(y_train_tl == 1)}")
    
    X_train_tl_tensor = torch.tensor(X_train_tl, dtype=torch.float32)
    y_train_tl_tensor = torch.tensor(y_train_tl, dtype=torch.float32)
    train_dataset_tl = TensorDataset(X_train_tl_tensor, y_train_tl_tensor)
    train_loader_tl = DataLoader(train_dataset_tl, batch_size=64, shuffle=True)
    
    tl_model = NNet(num_features)
    tl_model.train_model(train_loader_tl, num_epochs=10, learn_rate=0.001)
    results['tomek_links'] = tl_model.evaluate(test_loader)
    print("Tomek Links model evaluation:")
    for metric, value in results['tomek_links'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 7. NearMiss
    print("\n7. Training with NearMiss")
    nm = NearMiss(version=3)
    X_train_nm, y_train_nm = nm.fit_resample(X_train, y_train)
    print(f"After NearMiss - Class 0: {sum(y_train_nm == 0)}, Class 1: {sum(y_train_nm == 1)}")
    
    X_train_nm_tensor = torch.tensor(X_train_nm, dtype=torch.float32)
    y_train_nm_tensor = torch.tensor(y_train_nm, dtype=torch.float32)
    train_dataset_nm = TensorDataset(X_train_nm_tensor, y_train_nm_tensor)
    train_loader_nm = DataLoader(train_dataset_nm, batch_size=64, shuffle=True)
    
    nm_model = NNet(num_features)
    nm_model.train_model(train_loader_nm, num_epochs=10, learn_rate=0.001)
    results['nearmiss'] = nm_model.evaluate(test_loader)
    print("NearMiss model evaluation:")
    for metric, value in results['nearmiss'].items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Print summary of results
    print("\n=== Summary of Results ===")
    print("Method\t\tAccuracy\tPrecision\tRecall\t\tF1")
    for method, metrics in results.items():
        print(f"{method:15} {metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1']:.4f}")
    
    # Store class distribution data
    class_distribution = {
        'original': {'Class 0': num_zero, 'Class 1': num_one},
        'random_oversampling': {'Class 0': sum(y_train_ros == 0), 'Class 1': sum(y_train_ros == 1)},
        'smote': {'Class 0': sum(y_train_smote == 0), 'Class 1': sum(y_train_smote == 1)},
        'adasyn': {'Class 0': sum(y_train_adasyn == 0), 'Class 1': sum(y_train_adasyn == 1)},
        'tomek_links': {'Class 0': sum(y_train_tl == 0), 'Class 1': sum(y_train_tl == 1)},
        'nearmiss': {'Class 0': sum(y_train_nm == 0), 'Class 1': sum(y_train_nm == 1)}
    }
    
    # Generate plots
    print("\nGenerating plots...")
    import matplotlib.pyplot as plt
    import os
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Plot confusion matrices for all methods - ESSENTIAL
    def plot_confusion_matrices():
        methods = list(results.keys())
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, method in enumerate(methods):
            if i < len(axes):
                cm = results[method]['confusion_matrix']
                
                # Create confusion matrix
                cm_display = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])
                
                # Calculate metrics for title
                precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0
                recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Plot confusion matrix
                im = axes[i].imshow(cm_display, cmap='Blues')
                
                # Add text annotations
                for j in range(2):
                    for k in range(2):
                        axes[i].text(k, j, cm_display[j, k], ha='center', va='center', 
                                    color='black' if cm_display[j, k] < 400 else 'white', fontsize=12)
                
                # Set labels with method name and metrics
                method_name = method.replace('_', ' ').capitalize()
                axes[i].set_title(f"{method_name}\nF1: {f1:.3f}, Recall: {recall:.3f}", fontsize=12)
                axes[i].set_xticks([0, 1])
                axes[i].set_yticks([0, 1])
                axes[i].set_xticklabels(['Predicted 0', 'Predicted 1'])
                axes[i].set_yticklabels(['Actual 0', 'Actual 1'])
        
        # Remove empty subplots
        for i in range(len(methods), len(axes)):
            fig.delaxes(axes[i])
        
        # Add a title for the entire figure
        fig.suptitle('Confusion Matrices for Different Class Imbalance Handling Methods', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/confusion_matrices.png', dpi=300)
        plt.close()
        print("✓ Confusion matrices plot saved")

    # 2. Plot F1 vs Recall - ESSENTIAL for imbalanced classification
    def plot_f1_vs_recall():
        methods = list(results.keys())
        
        # Get F1 and recall values
        f1_scores = [results[method]['f1'] for method in methods]
        recall_scores = [results[method]['recall'] for method in methods]
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot points for each method
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            plt.scatter(recall_scores[i], f1_scores[i], color=colors[i], s=150, 
                       label=method.replace('_', ' ').capitalize())
            # Add method name as annotation
            plt.annotate(method.replace('_', ' ').capitalize(), 
                        (recall_scores[i], f1_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Add labels and title
        plt.xlabel('Recall (True Positive Rate)', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('F1 Score vs Recall for Different Class Imbalance Handling Methods', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Set axis limits with some padding
        plt.xlim([min(recall_scores)*0.9, max(recall_scores)*1.05])
        plt.ylim([min(f1_scores)*0.9, max(f1_scores)*1.05])
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/f1_vs_recall.png', dpi=300)
        plt.close()
        print("✓ F1 vs Recall plot saved")

    # 3. Plot class distribution comparison - ESSENTIAL to understand resampling effects
    def plot_class_distribution_comparison():
        methods = list(class_distribution.keys())
        
        # Create a figure
        plt.figure(figsize=(14, 8))
        
        # Set width of bars
        bar_width = 0.35
        index = np.arange(len(methods))
        
        # Get class counts
        class_0_counts = [class_distribution[method]['Class 0'] for method in methods]
        class_1_counts = [class_distribution[method]['Class 1'] for method in methods]
        
        # Calculate ratios for annotations
        ratios = [class_1/class_0 if class_0 > 0 else 0 for class_0, class_1 in zip(class_0_counts, class_1_counts)]
        
        # Plot bars
        plt.bar(index, class_0_counts, bar_width, label='Class 0 (Majority)', color='#1f77b4', alpha=0.8)
        plt.bar(index + bar_width, class_1_counts, bar_width, label='Class 1 (Minority)', color='#ff7f0e', alpha=0.8)
        
        # Add ratio annotations
        for i, ratio in enumerate(ratios):
            plt.text(i + bar_width/2, max(class_0_counts[i], class_1_counts[i]) + 20, 
                    f'Ratio: {ratio:.2f}', ha='center', fontsize=10)
        
        # Add labels and title
        plt.xlabel('Resampling Method', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution Before and After Resampling', fontsize=14)
        plt.xticks(index + bar_width/2, [m.replace('_', ' ').capitalize() for m in methods])
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{plots_dir}/class_distribution_comparison.png', dpi=300)
        plt.close()
        print("✓ Class distribution comparison plot saved")

    # Generate essential plots
    plot_confusion_matrices()
    plot_f1_vs_recall()
    plot_class_distribution_comparison()
    
    # Sort methods by F1 score
    print("\nBest performing methods based on F1 score:")
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (method, metrics) in enumerate(sorted_methods[:3]):
        print(f"{i+1}. {method.replace('_', ' ').capitalize()}: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}")
    
