import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution with kernel size 1
        other = 0.5  # Define 'other' as a scalar
        t2 = t1 - other  # Subtract 'other' from the output of the convolution
        return t2

# Initialize the model
model = Model()

# Generate an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 color channels, 64x64 image size

# Get the output of the model
output = model(input_tensor)

# Print the output shape
print(output.shape)
