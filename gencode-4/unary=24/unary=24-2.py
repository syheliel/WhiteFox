import torch

# Define the model
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution with kernel size 1 to the input tensor
        t2 = t1 > 0  # Create a boolean mask where each element is True if t1 > 0
        t3 = t1 * self.negative_slope  # Multiply the output of the convolution by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the where function to select elements from t1 or t3 based on the mask t2
        return t4

# Initialize the model
model = LeakyReLUModel()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Get the output of the model
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)
