import torch

class CustomCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target): # ctx = context
        # computes -log(softmax(logits)) for the correct class
        # logits = tensor of shape (batch_size, num_classes) containing features
        # target: tensor of shape (batch_size), comtaining class labels

        batch_size, num_classes = logits.shape

        # apply softmax go get predicted probablities
        softmax_probs = torch.softmax(logits, dim = 1)

        # determine the predicted probability of the correct class
        correct_class_probs = softmax_probs[torch.arange(batch_size), target]

        # compute NLL (Negative Log Likelihood)
        loss = -torch.log(correct_class_probs + 1e-9).mean() # we add a small amount to avoid log(0)

        # save tensors for backward propagation
        ctx.save_for_backward(softmax_probs, target)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # compute gradients of cross-entropy loss

        softmax_probs, target = ctx.saved_tensors
        batch_size = target.shape[0]

        # create one-hot encoding for target classes
        one_hot_target = torch.zeros_like(softmax_probs)
        one_hot_target[torch.arange(batch_size), target] = 1

        # compute gradient and clip it to [-1, 1]
        grad_logits = (softmax_probs - one_hot_target) / batch_size
        grad_logits = torch.clamp(grad_logits, -1, 1)  # clip gradients

        # scale the gradient by grad_output
        return grad_logits * grad_output, None # no gradient for target


