import torch
import matplotlib.pyplot as plt

def run_task1():
    print("\nRunning task 1")

    def f(x,y):
        return 3*x**3 - y**2

    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([1.0], requires_grad=True)
    z = f(x,y)
    z.backward()
    print(f"dz/dx: {x.grad}")
    print(f"dz/dy: {y.grad}")

def run_task2():
    print("\nRunning task 2")
    def f2(x):
        return torch.sin(x) * torch.exp(x)
    
    def analytical_grad(x):
        return torch.exp(x) * (torch.sin(x) + torch.cos(x))
    
    # Generate test points
    x_vals = torch.linspace(-2, 2, 100, dtype=torch.float64)
    grad_analytical = analytical_grad(x_vals)
    
    # Autograd gradients
    x_vals_autograd = x_vals.clone().requires_grad_()
    y = f2(x_vals_autograd)
    y.sum().backward()  # Sum to create a scalar for backward
    grad_autograd = x_vals_autograd.grad
    
    # Central difference
    h = 1e-4
    grad_central = (f2(x_vals + h) - f2(x_vals - h)) / (2 * h)
    
    # RMSE calculations
    rmse_autograd = torch.sqrt(torch.mean((grad_autograd - grad_analytical)**2))
    rmse_central = torch.sqrt(torch.mean((grad_central - grad_analytical)**2))
    
    print(f"Analytical vs Autograd RMSE: {rmse_autograd.item():.4e}")
    print(f"Analytical vs Central RMSE: {rmse_central.item():.4e}")
    
def run_task3():
    print("\nRunning task 3")
    from utils import generate_function, gradient_descent    

    # Parameters
    learning_rate = 0.01
    num_iterations = 80

    # Test different dimensionalities and complexity levels
    dimensionalities = [2, 3, 5, 7]
    complexity_levels = [1, 2, 3]

    loss_histories = {}

    for complexity in complexity_levels:
        for dim in dimensionalities:
            print(f"\nMinimizing function with complexity={complexity}, dimensionality={dim}")

            # Generate the function
            func = generate_function(dim, complexity)

            # Initialize a random starting point
            initial_point = torch.randn(dim, dtype=torch.float64)

            # Perform gradient descent
            loss_history = gradient_descent(func, initial_point, learning_rate, num_iterations)

            # Store loss history for plotting
            loss_histories[(dim, complexity)] = loss_history

            # Print final loss
            print(f"Final loss: {loss_history[-1]:.6f}")

    # Plot all loss histories together
    plt.figure(figsize=(14, 8))

    # Plot by dimensionality
    for i, dim in enumerate(dimensionalities):
        plt.subplot(2, 2, i + 1)
        for complexity in complexity_levels:
            plt.plot(loss_histories[(dim, complexity)], label=f"Complexity={complexity}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Dimensionality={dim}")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig("task3_dimensionality.png")

    # Plot by complexity
    plt.figure(figsize=(26/1.5, 8/1.5))
    for i, complexity in enumerate(complexity_levels):
        plt.subplot(1, 3, i + 1)
        for dim in dimensionalities:
            plt.plot(loss_histories[(dim, complexity)], label=f"Dimensionality={dim}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"Complexity={complexity}")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig("task3_complexity.png")

def run_task4():
    print("\nRunning task 4")

    from neuralnet_grad import NNet
    from neuralnet_grad import generate_dataset

    num_instances = 100
    num_features = 10
    num_classes = 6
    data_noise = 0.5
    X, y = generate_dataset(num_instances, num_features, num_classes, data_noise)

    trainSplit = 0.8
    num_train_instances = int(num_instances * trainSplit)

    X_train, y_train = X[:num_train_instances], y[:num_train_instances]
    X_test, y_test = X[num_train_instances:], y[num_train_instances:]

    
    learn_rate = 0.01
    num_epochs = 1000

    print("Training model with standard gradients")
    model = NNet(num_features, num_classes)
    model.train_model(X_train, y_train, learn_rate, num_epochs)

    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)

    print(f'\tTrain accuracy with standard gradients: {train_acc:.4f}') 
    print(f'\tTest accuracy with standard gradients: {test_acc  :.4f}')

    
    print("Training model with explicit gradients")
    model_explicit = NNet(num_features, num_classes)
    model_explicit.train_model_explicit(X_train, y_train, learn_rate, num_epochs)

    train_acc = model_explicit.evaluate(X_train, y_train)
    test_acc = model_explicit.evaluate(X_test, y_test)

    print(f'\tTrain accuracy with explicit gradients: {train_acc:.4f}')
    print(f'\tTest accuracy with explicit gradients: {test_acc:.4f}')


    print("\nTraining model with momentum")
    model_momentum = NNet(num_features, num_classes)
    model_momentum.train_model_momentum(X_train, y_train, learn_rate, num_epochs)

    train_acc = model_momentum.evaluate(X_train, y_train)
    test_acc = model_momentum.evaluate(X_test, y_test)

    print(f'\tTrain accuracy with momentum: {train_acc:.4f}')
    print(f'\tTest accuracy with momentum: {test_acc:.4f}')


    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history, label='Standard Gradients', linestyle='--', alpha=0.8)
    plt.plot(model_explicit.loss_history, label='Explicit Gradients', linestyle='--', alpha=0.8)
    plt.plot(model_momentum.loss_history, label='Momentum', linestyle='-.', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Loss Evolution")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(model.accuracy_history, label='Standard Gradients', linestyle='--', alpha=0.8)
    plt.plot(model_explicit.accuracy_history, label='Explicit Gradients', linestyle='--', alpha=0.8)
    plt.plot(model_momentum.accuracy_history, label='Momentum', linestyle='-.', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Accuracy Evolution")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("task4_training_progress.png")
    plt.close()
    
def run_task5():
    print("\nRunning task 5")
    from custom_autograd_functions import CustomCrossEntropyLoss
    from neuralnet_grad import NNet, generate_dataset

    num_instances = 100
    num_features = 10
    num_classes = 6
    data_noise = 0.5
    X, y = generate_dataset(num_instances, num_features, num_classes, data_noise)

    trainSplit = 0.8
    num_train_instances = int(num_instances * trainSplit)
    X_train, y_train = X[:num_train_instances], y[:num_train_instances]
    X_test, y_test = X[num_train_instances:], y[num_train_instances:]

    learn_rate = 0.01
    num_epochs = 450

    # Train with standard cross-entropy loss
    model_standard = NNet(num_features, num_classes)
    model_standard.train_model_momentum(X_train, y_train, learn_rate, num_epochs)

    train_acc = model_standard.evaluate(X_train, y_train)
    test_acc = model_standard.evaluate(X_test, y_test)

    print(f'Train accuracy with standard cross-entropy loss: {train_acc:.4f}')
    print(f'Test accuracy with standard cross-entropy loss: {test_acc:.4f}')

    # Train with custom cross-entropy loss
    model_ce = NNet(num_features, num_classes)
    model_ce.train_model_momentum(X_train, y_train, learn_rate, num_epochs, lossFunc=CustomCrossEntropyLoss)

    train_acc = model_ce.evaluate(X_train, y_train)
    test_acc = model_ce.evaluate(X_test, y_test)

    print(f'Train accuracy with custom cross-entropy loss: {train_acc:.4f}')
    print(f'Test accuracy with custom cross-entropy loss: {test_acc:.4f}')

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model_standard.loss_history, label='Standard CE Loss', linestyle='--', alpha=0.8)
    plt.plot(model_ce.loss_history, label='Custom CE Loss', linestyle='-.', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Loss Evolution")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(model_standard.accuracy_history, label='Standard CE Loss', linestyle='--', alpha=0.8)
    plt.plot(model_ce.accuracy_history, label='Custom CE Loss', linestyle='-.', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Accuracy Evolution")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("task5_custom_cross_entropy.png")
    plt.close() 

def run_task6():
    print("\nRunning task 6")
    from neuralnet_grad import NNet, generate_dataset

    num_instances = 100
    num_features = 10
    num_classes = 6
    data_noise = 0.5
    X, y = generate_dataset(num_instances, num_features, num_classes, data_noise)

    trainSplit = 0.8
    num_train_instances = int(num_instances * trainSplit)
    X_train, y_train = X[:num_train_instances], y[:num_train_instances]
    X_test, y_test = X[num_train_instances:], y[num_train_instances:]

    learn_rate = 0.01
    num_epochs = 1000

    # Train with standard Cross-Entropy loss
    model_ce = NNet(num_features, num_classes)
    model_ce.train_model_momentum(X_train, y_train, learn_rate, num_epochs)

    train_acc_ce = model_ce.evaluate(X_train, y_train)
    test_acc_ce = model_ce.evaluate(X_test, y_test)

    print(f'Train accuracy with standard Cross-Entropy loss: {train_acc_ce:.4f}')
    print(f'Test accuracy with standard Cross-Entropy loss: {test_acc_ce:.4f}')

    # Train with custom MSE loss
    model_mse = NNet(num_features, num_classes)
    model_mse.train_model_momentum(X_train, y_train, learn_rate, num_epochs, lossFunc=torch.nn.MSELoss())

    train_acc_mse = model_mse.evaluate(X_train, y_train)
    test_acc_mse = model_mse.evaluate(X_test, y_test)

    print(f'Train accuracy with custom MSE loss: {train_acc_mse:.4f}')
    print(f'Test accuracy with custom MSE loss: {test_acc_mse:.4f}')

    # Plot results
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model_ce.loss_history, label='Standard CE Loss', linestyle='--', alpha=0.8)
    plt.plot(model_mse.loss_history, label='Custom MSE Loss', linestyle='-.', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Loss Evolution")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(model_ce.accuracy_history, label='Standard CE Loss', linestyle='--', alpha=0.8)
    plt.plot(model_mse.accuracy_history, label='Custom MSE Loss', linestyle='-.', alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Accuracy Evolution")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("task6_custom_mse.png")
    plt.close()

    # save the csv file
    import pandas as pd

    # save progress
    df = pd.DataFrame({
        'Epoch': range(num_epochs),
        'Standard CE Loss': model_ce.loss_history,
        'Custom MSE Loss': model_mse.loss_history,
        'Standard CE Accuracy': model_ce.accuracy_history,
        'Custom MSE Accuracy': model_mse.accuracy_history
    })

    df.to_csv("task6_results.csv", index=False)



if __name__ == "__main__":
    #run_task1()
    #run_task2()
    #run_task3()
    #run_task4()
    #run_task5()
    run_task6()