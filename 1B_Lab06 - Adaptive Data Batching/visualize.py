import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import argparse
import sys

def load_experiment_data(task_id):
    """Load data for a specific task"""
    results_dir = Path("results")
    file_path = results_dir / f"{task_id}_metrics.csv"
    
    if not file_path.exists():
        print(f"No data found for task {task_id} at {file_path}")
        return None
    
    try:
        df = pl.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data for task {task_id}: {e}")
        return None

def plot_experiment_metrics(task_id, title=None):
    """Plot metrics for a single experiment"""
    df = load_experiment_data(task_id)
    if df is None:
        return
    
    if title is None:
        title = f"Task {task_id}"
    
    # Convert DataFrame to pandas for easier plotting with matplotlib
    df_pd = df.to_pandas()
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Loss vs Epoch
    axes[0, 0].plot(df_pd['epoch'], df_pd['train_loss'], 'b-o', markersize=4, label='Train')
    axes[0, 0].plot(df_pd['epoch'], df_pd['test_loss'], 'r-o', markersize=4, label='Test')
    axes[0, 0].set_title('Loss vs Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Epoch
    axes[0, 1].plot(df_pd['epoch'], df_pd['train_accuracy'], 'g-o', markersize=4, label='Train')
    axes[0, 1].plot(df_pd['epoch'], df_pd['test_accuracy'], 'r-o', markersize=4, label='Test')
    axes[0, 1].set_title('Accuracy vs Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Batch Size vs Epoch
    axes[1, 0].plot(df_pd['epoch'], df_pd['batch_size'], 'c-o', markersize=4)
    axes[1, 0].set_title('Batch Size vs Epoch')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Batch Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Batch Size (scatter plot)
    scatter = axes[1, 1].scatter(df_pd['batch_size'], df_pd['test_accuracy'], 
                                  c=df_pd['epoch'], cmap='viridis', 
                                  s=50, alpha=0.7)
    axes[1, 1].set_title('Test Accuracy vs Batch Size')
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Epoch')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    results_dir = Path("results")
    fig.savefig(results_dir / f"{task_id}_plot.png", dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def compare_experiments_by_metric(task_ids, metric='test_accuracy', title=None):
    """Compare a specific metric across multiple experiments"""
    if title is None:
        title = f"Comparison of {metric} across tasks"
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for task_id in task_ids:
        df = load_experiment_data(task_id)
        if df is None:
            continue
        
        df_pd = df.to_pandas()
        ax.plot(df_pd['epoch'], df_pd[metric], '-o', markersize=4, label=f"{task_id}")
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path("results")
    fig.savefig(results_dir / f"comparison_{metric}.png", dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def list_available_tasks():
    """List all available task results in the results directory"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found")
        return []
        
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    task_ids = [f.replace('_metrics.csv', '') for f in csv_files]
    
    if not task_ids:
        print("No task results found")
    else:
        print(f"Available tasks ({len(task_ids)}):")
        for task_id in sorted(task_ids):
            print(f"  - {task_id}")
    
    return task_ids

def compare_datasets_by_strategy(strategy_name, metric='test_accuracy'):
    """Compare how a specific strategy performs across different datasets"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found")
        return
        
    # Find all task IDs that contain the strategy name
    task_ids = []
    for f in os.listdir(results_dir):
        if f.endswith('_metrics.csv') and strategy_name.lower() in f.lower():
            task_id = f.replace('_metrics.csv', '')
            task_ids.append(task_id)
    
    if not task_ids:
        print(f"No tasks found with strategy '{strategy_name}'")
        return
    
    title = f"Comparison of {strategy_name} strategy across datasets ({metric})"
    return compare_experiments_by_metric(task_ids, metric, title)

def find_related_tasks(task_id, all_tasks):
    """Find tasks related to the given task (same dataset or strategy)"""
    # Extract components from task_id: exercise, dataset, strategy
    match = re.match(r'(\d+\.\d+)_(\w+)_(\w+)', task_id)
    if not match:
        return []
    
    _, dataset, strategy = match.groups()
    
    # Find related by dataset
    dataset_related = [t for t in all_tasks if f"_{dataset}_" in t and t != task_id]
    
    # Find related by strategy
    strategy_related = [t for t in all_tasks if f"_{strategy}" in t and t != task_id]
    
    return list(set(dataset_related + strategy_related))

def plot_all_results(output_dir='plots', metric='test_accuracy', save_only=False):
    """Plot all results found in the results directory"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all available tasks
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found")
        return
    
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    task_ids = [f.replace('_metrics.csv', '') for f in csv_files]
    
    if not task_ids:
        print("No task results found")
        return
    
    print(f"Found {len(task_ids)} tasks in the results folder")
    
    # Group tasks by dataset + strategy pattern
    task_patterns = {}
    for task_id in task_ids:
        # Group by exercise
        match = re.match(r'(\d+\.\d+)', task_id)
        if match:
            exercise = match.group(1)
            if exercise not in task_patterns:
                task_patterns[exercise] = []
            task_patterns[exercise].append(task_id)
    
    # Plot each task
    for exercise, tasks in task_patterns.items():
        print(f"\nPlotting exercise {exercise} tasks...")
        for task_id in tasks:
            print(f"  - Plotting {task_id}")
            fig = plot_experiment_metrics(task_id)
            if fig and save_only:
                fig_path = output_dir / f"{task_id}_plot.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    # Also generate comparison plots by dataset
    print("\nGenerating dataset comparison plots...")
    datasets = ['iris', 'fashion', 'custom']
    
    for dataset in datasets:
        dataset_tasks = [t for t in task_ids if f"_{dataset}_" in t]
        if dataset_tasks:
            print(f"  - Comparing strategies for {dataset} dataset")
            fig = compare_experiments_by_metric(
                dataset_tasks, 
                metric,
                f"Comparison of strategies for {dataset} dataset ({metric})"
            )
            if fig and save_only:
                fig_path = output_dir / f"comparison_{dataset}_{metric}.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
    
    print("\nAll plots generated successfully!")

def plot_by_strategy(metric='test_accuracy'):
    """Compare strategies across different datasets"""
    # Find all available tasks
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found")
        return
    
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    task_ids = [f.replace('_metrics.csv', '') for f in csv_files]
    
    if not task_ids:
        print("No task results found")
        return
    
    # Extract unique strategies
    strategies = set()
    for task_id in task_ids:
        match = re.search(r'_(\w+)$', task_id)
        if match:
            strategies.add(match.group(1))
    
    print(f"Comparing across datasets for strategies: {', '.join(strategies)}")
    for strategy in strategies:
        print(f"Comparing {strategy} strategy across datasets")
        compare_datasets_by_strategy(strategy, metric)

def plot_by_dataset(metric='test_accuracy'):
    """Compare different strategies for each dataset"""
    # Find all available tasks
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found")
        return
    
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_metrics.csv')]
    task_ids = [f.replace('_metrics.csv', '') for f in csv_files]
    
    if not task_ids:
        print("No task results found")
        return
    
    # Extract unique datasets
    datasets = set()
    for task_id in task_ids:
        match = re.search(r'_(\w+)_', task_id)
        if match:
            datasets.add(match.group(1))
    
    print(f"Comparing strategies for datasets: {', '.join(datasets)}")
    for dataset in datasets:
        dataset_tasks = [t for t in task_ids if f"_{dataset}_" in t]
        if dataset_tasks:
            print(f"Comparing strategies for {dataset} dataset")
            compare_experiments_by_metric(
                dataset_tasks, 
                metric,
                f"Comparison of strategies for {dataset} dataset ({metric})"
            )

def main():
    parser = argparse.ArgumentParser(description='Plot results for tasks')
    parser.add_argument('--task', help='Specific task ID to plot')
    parser.add_argument('--compare', choices=['dataset', 'strategy'], 
                        help='Compare results across datasets or strategies')
    parser.add_argument('--metric', default='test_accuracy', 
                        help='Metric to compare (default: test_accuracy)')
    parser.add_argument('--save-only', action='store_true', 
                        help='Save plots without displaying them')
    parser.add_argument('--output-dir', default='plots',
                        help='Directory to save plots (default: plots)')
    args = parser.parse_args()

    # Handle specific task plotting
    if args.task:
        task_ids = list_available_tasks()
        if args.task not in task_ids:
            print(f"Task '{args.task}' not found")
            return
            
        print(f"Plotting metrics for task '{args.task}'")
        fig = plot_experiment_metrics(args.task)
        if fig and args.save_only:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            fig_path = output_dir / f"{args.task}_plot.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved to {fig_path}")
        return
        
    # Handle comparison
    if args.compare:
        if args.compare == 'dataset':
            plot_by_strategy(args.metric)
        elif args.compare == 'strategy':
            plot_by_dataset(args.metric)
        return
    
    # If no specific arguments, plot everything
    plot_all_results(args.output_dir, args.metric, args.save_only)

if __name__ == "__main__":    
    if len(sys.argv) > 1:
        main()