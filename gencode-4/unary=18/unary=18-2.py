import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Apply sigmoid activation function
        t2 = torch.sigmoid(t1)
        return t2

# Initializing the model
model = SimpleModel()

# Input tensor: Batch size of 1, 3 channels, 64x64 image
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)
