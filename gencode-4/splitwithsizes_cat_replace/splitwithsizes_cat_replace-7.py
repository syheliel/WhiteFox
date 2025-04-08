import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolution layer with 3 input channels, 8 output channels, and a 3x3 kernel
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Define an attribute for the other tensor
        self.scalar_addition = torch.tensor(2.0, dtype=torch.float32)  # A scalar value to add

    def forward(self, x):
        # Apply convolution to the input tensor
        conv_output = self.conv(x)
        # Add the scalar to the output of the convolution
        output = conv_output + self.scalar_addition
        return output

# Initializing the model
model = CustomModel()

# Generating input tensor for the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)
