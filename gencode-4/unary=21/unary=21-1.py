import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x1):
        t1 = self.conv(x1)          # Apply pointwise convolution
        t2 = torch.tanh(t1)         # Apply hyperbolic tangent activation function
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
__output__ = m(x1)               # Forward pass through the model

# Output shape
print(__output__.shape)
