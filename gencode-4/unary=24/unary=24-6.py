import torch

# Define the model
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 > 0  # Create a boolean mask
        t3 = t1 * self.negative_slope  # Multiply by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the where function
        return t4

# Initialize the model
model = LeakyReLUModel(negative_slope=0.01)

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

# Print the output shape
print(output.shape)  # Output shape should be (1, 8, 64, 64)
