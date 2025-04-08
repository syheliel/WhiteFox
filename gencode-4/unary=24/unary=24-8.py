import torch

# Model definition
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 > 0  # Create a boolean mask
        t3 = t1 * self.negative_slope  # Multiply the output of the convolution by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the where function
        return t4

# Initializing the model
model = LeakyReLUModel(negative_slope=0.01)

# Generating input tensor (batch size 1, 3 channels, 64x64 image)
input_tensor = torch.randn(1, 3, 64, 64)

# Running the model with the input tensor
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)
