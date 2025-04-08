import torch

class SubtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution: from 3 input channels to 5 output channels with kernel size 1
        self.conv = torch.nn.Conv2d(3, 5, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Subtract a scalar value (e.g., 0.3)
        t2 = t1 - 0.3
        return t2

# Initializing the model
model = SubtractionModel()

# Generating input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Output the shape of the output tensor
print("Output shape:", output.shape)

input_tensor = torch.randn(1, 3, 64, 64)
