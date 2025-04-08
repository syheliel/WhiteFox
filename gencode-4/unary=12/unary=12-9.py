import torch

# Define the model
class GatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.sigmoid(t1)  # Apply sigmoid function
        t3 = t1 * t2  # Multiply convolution output by sigmoid output
        return t3

# Initializing the model
model = GatingModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

# Print the output shape to verify
print("Output shape:", output.shape)
