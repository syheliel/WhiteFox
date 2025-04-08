import torch

# Define the new model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with different input and output channels
        self.conv = torch.nn.Conv2d(5, 10, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Multiply the two outputs
        return t6

# Initialize the model
model = CustomModel()

# Generate input tensor with different dimensions
input_tensor = torch.randn(1, 5, 32, 32)  # Batch size of 1, 5 channels, 32x32 resolution

# Get the output from the model
output_tensor = model(input_tensor)

# Display the output tensor shape
print(output_tensor.shape)
