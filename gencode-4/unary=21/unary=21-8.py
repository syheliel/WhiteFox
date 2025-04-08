import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = torch.tanh(t1)  # Apply hyperbolic tangent activation
        return t2

# Initializing the model
m = Model()

# Input tensor for the model
x1 = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions

# Forward pass through the model
output = m(x1)
