import torch

# Define the model class
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with 3 input channels and 8 output channels
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 * 0.5                 # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)           # Apply the error function
        t5 = t4 + 1                   # Add 1 to the error function output
        t6 = t2 * t5                  # Multiply the results
        return t6

# Initialize the model
model = CustomModel()

# Generate an input tensor with batch size 1, 3 channels, and 64x64 dimensions
input_tensor = torch.randn(1, 3, 64, 64)

# Get the model's output
output = model(input_tensor)

# Display the output shape
print(output.shape)
