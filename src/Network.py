import torch

class SRFU(torch.nn.Module):
    def __init__(self, layers):
        self.layers = layers
        self.init_params()

    def init_params(self, std=0.1):
        self.weight = std*torch.randn_like(self.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x):
        grad = x.clone()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def optimizer_step(self, lr):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                layer.weight -= lr * layer.weight_grad
                layer.bias -= lr * layer.bias_grad
