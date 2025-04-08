import torch

# Define the model
class TanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Apply the hyperbolic tangent function
        t2 = torch.tanh(t1)
        return t2

# Initialize the model
model = TanhModel()

# Generate an input tensor
# Example input tensor with batch size 1, 3 channels (e.g., RGB), and 64x64 image size
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
