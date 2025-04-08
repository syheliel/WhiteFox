import torch

class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.tanh(t1)  # Apply hyperbolic tangent activation
        return t2

# Initializing the model
model = TanhModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor with batch size 1, 3 channels, and 64x64 size
output = model(input_tensor)

# Print the output shape
print(output.shape)
