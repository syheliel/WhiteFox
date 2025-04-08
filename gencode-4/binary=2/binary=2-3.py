import torch

class SubtractModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, other):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 - other      # Subtract 'other' from the output of the convolution
        return t2

# Initializing the model
model = SubtractModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
other_value = torch.tensor(0.5).view(1, 8, 1, 1)  # Scalar value to subtract, reshaped to match output shape

# Forward pass
output = model(input_tensor, other_value)

# Displaying the output shape
print(output.shape)  # Expected output shape is (1, 8, 64, 64)
