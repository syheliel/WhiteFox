import torch

class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.tanh(t1)  # Apply the hyperbolic tangent function
        return t2

# Initializing the model
model = TanhModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)
