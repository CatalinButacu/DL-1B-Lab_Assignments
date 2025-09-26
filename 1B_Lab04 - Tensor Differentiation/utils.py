import torch

def generate_function(num_dimensions, complexity_level = 1):
    def func(x):
        base = torch.sum(x**2)
        if complexity_level >= 2:
            base += torch.sum(torch.sin(3 * x))
        if complexity_level >= 3:
            base += torch.sum(x**4)
        return base
    return func


def gradient_descent(func, initial_point, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent on the given function.
    """
    x = initial_point.clone().requires_grad_()
    loss_history = []

    for i in range(num_iterations):
        # Compute the function value
        loss = func(x)
        loss_history.append(loss.item())

        # Compute gradients
        loss.backward()

        # Update parameters using gradient descent
        with torch.no_grad():
            x -= learning_rate * x.grad

        # Zero gradients for the next iteration
        x.grad.zero_()

    return loss_history
