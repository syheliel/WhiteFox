import torch

class SubtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        # Define a scalar 'other' to subtract from the convolution output
        self.other = 0.5

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 - self.other  # Subtract 'other' from the convolution output
        return t2

# Initializing the model
model = SubtractionModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

print("Output shape:", output.shape)
