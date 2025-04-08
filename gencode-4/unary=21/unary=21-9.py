import torch

# Model Definition
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)          # Apply the convolution
        t2 = torch.tanh(t1)       # Apply the hyperbolic tangent activation
        return t2

# Initializing the model
model = TanhModel()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Forward pass through the model
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)
