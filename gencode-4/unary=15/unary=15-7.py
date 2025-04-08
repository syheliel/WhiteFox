import torch

# Define the model
class ReLUConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Apply ReLU activation function
        t2 = torch.nn.functional.relu(t1)
        return t2

# Initializing the model
model = ReLUConvModel()

# Create input tensor with batch size 1, 3 channels, and 64x64 dimensions
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output of the model
output = model(input_tensor)

# Display the output shape
print(output.shape)
