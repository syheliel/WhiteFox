import torch

# Define the model
class SubtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)  # Pointwise convolution with kernel size 1

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply convolution
        other = 0.5  # Subtract scalar value
        t2 = t1 - other  # Subtract 'other' from the output of the convolution
        return t2

# Initialize the model
model = SubtractionModel()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions

# Get the output from the model
output = model(input_tensor)

# Print the output shape to verify
print(output.shape)
