import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x1, other):
        t1 = self.conv(x1)         # Apply pointwise convolution
        t2 = t1 + other            # Add another tensor to the output of the convolution
        t3 = torch.nn.functional.relu(t2)  # Apply ReLU activation
        return t3

# Initializing the model
model = CustomModel()

# Create an input tensor of shape (1, 3, 64, 64)
input_tensor = torch.randn(1, 3, 64, 64)

# Create another tensor to add (should match the output shape of the convolution)
other_tensor = torch.randn(1, 8, 64, 64)  # Shape should be (batch_size, out_channels, height, width)

# Get the output from the model
output = model(input_tensor, other_tensor)

print(output.shape)  # Output shape
