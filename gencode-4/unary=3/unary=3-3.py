import torch

# Define the model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Pointwise convolution layer with kernel size 1
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Changed output channels to 16

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1 to the error function output
        t6 = t2 * t5  # Multiply the scaled output by the modified error function output
        return t6

# Initialize the model
model = CustomModel()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Forward pass
output = model(input_tensor)

# Print the output shape
print(output.shape)
