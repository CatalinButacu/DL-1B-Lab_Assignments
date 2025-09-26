import os 
import time
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, ConcatDataset

from shapes_dataset import ShapesDataset, show_images
from static_cnn import StaticCNN
from dynamic_cnn import DynamicCNN

def plot_training_history(history, title):
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        fig, ax1 = plt.subplots(figsize=(10, 4))

        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)

        if 'overall_loss' in history: # dynamic CNN
            ax1.plot(history['overall_epochs'], history['overall_loss'], color=color, linestyle='-', label='Overall Loss')
            epochs_list = history['overall_epochs']
        elif 'loss' in history: # static CNN
            ax1.plot(history['loss'], color=color, linestyle='-', label='Loss')
            epochs_list = range(1, len(history['loss']) + 1)

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy (%)', color=color)  
        if 'overall_train_accuracy' in history: # dynamic CNN
            ax2.plot(epochs_list, history['overall_train_accuracy'], color=color, linestyle='--', label='Overall Train Accuracy')
        elif 'accuracy' in history: # static CNN
            ax2.plot(epochs_list, history['accuracy'], color=color, linestyle='--', label='Train Accuracy')

        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.7)
        fig.tight_layout() 
        plt.title(title)
        
        file_name = title.replace(' ', '_').lower() + '_history.png'
        file_path = os.path.join(plots_dir, file_name)
        plt.savefig(file_path)
        plt.close(fig)


if __name__ == '__main__':
    torch.manual_seed(42)
    IMAGE_SIZE = 64
    NUM_CLASSES = 3
    BATCH_SIZE = 64
    NUM_INSTANCES_PER_SET = 1000
    EPOCHS_STATIC = 60
    EPOCHS_PER_STAGE_DYNAMIC = EPOCHS_STATIC // NUM_CLASSES
    LEARN_RATE = 0.001

    print(f"Using {NUM_INSTANCES_PER_SET} instances per dataset difficulty.")

    print("\n--- Preparing Datasets ---")
    easy_train_dataset = ShapesDataset(stage='easy', num_instances=NUM_INSTANCES_PER_SET, image_size=IMAGE_SIZE, random_seed=42)
    medium_train_dataset = ShapesDataset(stage='medium', num_instances=NUM_INSTANCES_PER_SET, image_size=IMAGE_SIZE, random_seed=42)
    hard_train_dataset = ShapesDataset(stage='hard', num_instances=NUM_INSTANCES_PER_SET, image_size=IMAGE_SIZE, random_seed=42)
    
    easy_test_dataset = ShapesDataset(stage='easy', num_instances=NUM_INSTANCES_PER_SET // 2, image_size=IMAGE_SIZE, random_seed=123)
    medium_test_dataset = ShapesDataset(stage='medium', num_instances=NUM_INSTANCES_PER_SET // 2, image_size=IMAGE_SIZE, random_seed=123)
    hard_test_dataset = ShapesDataset(stage='hard', num_instances=NUM_INSTANCES_PER_SET // 2, image_size=IMAGE_SIZE, random_seed=123)

    all_train_dataset = ConcatDataset([easy_train_dataset, medium_train_dataset, hard_train_dataset])
    all_test_dataset = ConcatDataset([easy_test_dataset, medium_test_dataset, hard_test_dataset])

    # Traaining DataLoaders
    easy_train_loader = DataLoader(easy_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    medium_train_loader = DataLoader(medium_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    hard_train_loader = DataLoader(hard_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    all_train_loader = DataLoader(all_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Testing DataLoaders
    easy_test_loader = DataLoader(easy_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    medium_test_loader = DataLoader(medium_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    hard_test_loader = DataLoader(hard_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_test_loader = DataLoader(all_test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    print("\n--- Visualizing Dataset Samples ---")
    show_images(easy_train_dataset, medium_train_dataset, hard_train_dataset, images_per_stage=3)


    print("\n--- Training Static CNN ---")
    static_model = StaticCNN(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
    start_time_static = time.time()
    static_history = static_model.train_model(all_train_loader, num_epochs=EPOCHS_STATIC, learn_rate=LEARN_RATE)
    training_time_static = time.time() - start_time_static
    print(f"Static CNN Training Time: {training_time_static:.2f} seconds")

    print("\n--- Evaluating Static CNN ---")
    static_acc_easy = static_model.evaluate_model(easy_test_loader)
    static_acc_medium = static_model.evaluate_model(medium_test_loader)
    static_acc_hard = static_model.evaluate_model(hard_test_loader)
    static_acc_all = static_model.evaluate_model(all_test_loader)
    print(f'Static CNN Accuracy: Easy={static_acc_easy:.2f}%, Medium={static_acc_medium:.2f}%, Hard={static_acc_hard:.2f}%, Overall={static_acc_all:.2f}%')


    print("\n--- Training Dynamic CNN ---")
    dynamic_model = DynamicCNN(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
    start_time_dynamic = time.time()
    dynamic_history = dynamic_model.train_model(easy_train_loader, medium_train_loader, hard_train_loader, 
                                              num_epochs_per_stage=int(EPOCHS_PER_STAGE_DYNAMIC), learn_rate=LEARN_RATE)
    training_time_dynamic = time.time() - start_time_dynamic
    print(f"Dynamic CNN Training Time: {training_time_dynamic:.2f} seconds")

    print("\n--- Evaluating Dynamic CNN ---")
    if dynamic_model.current_stage < 3:
        print("Warning: Dynamic model not fully complex for final evaluation. Manually setting to stage 3.")
        if dynamic_model.current_stage < 2: dynamic_model._increase_complexity_stage2()
        if dynamic_model.current_stage < 3: dynamic_model._increase_complexity_stage3()

    dynamic_acc_easy = dynamic_model.evaluate_model(easy_test_loader)
    dynamic_acc_medium = dynamic_model.evaluate_model(medium_test_loader)
    dynamic_acc_hard = dynamic_model.evaluate_model(hard_test_loader)
    dynamic_acc_all = dynamic_model.evaluate_model(all_test_loader)
    print(f'Dynamic CNN Accuracy: Easy={dynamic_acc_easy:.2f}%, Medium={dynamic_acc_medium:.2f}%, Hard={dynamic_acc_hard:.2f}%, Overall={dynamic_acc_all:.2f}%')


    print("\n--- Final Results ---")
    print(f"Training Time:")
    print(f"  Static CNN: {training_time_static:.2f} seconds")
    print(f"  Dynamic CNN: {training_time_dynamic:.2f} seconds")

    print(f"\nAccuracy on Test Data:")
    print(f"  Difficulty | Static CNN | Dynamic CNN")
    print(f"  --------------------------------------")
    print(f"  Easy       | {static_acc_easy:>8.2f}% | {dynamic_acc_easy:>9.2f}%")
    print(f"  Medium     | {static_acc_medium:>8.2f}% | {dynamic_acc_medium:>9.2f}%")
    print(f"  Hard       | {static_acc_hard:>8.2f}% | {dynamic_acc_hard:>9.2f}%")
    print(f"  --------------------------------------")
    print(f"  Overall    | {static_acc_all:>8.2f}% | {dynamic_acc_all:>9.2f}%")


    print("\n--- Plotting Training History ---")
    plot_training_history(static_history, "Static CNN Training Progress")
    plot_training_history(dynamic_history, "Dynamic CNN Training Progress")

