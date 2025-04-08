import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Using a pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)

    def forward(self, x1):
        v1 = self.conv(x1)  # Apply pointwise convolution
        v2 = v1 * 0.5  # Multiply by 0.5
        v3 = v1 * 0.7071067811865476  # Multiply by 0.707...
        v4 = torch.erf(v3)  # Apply the error function
        v5 = v4 + 1  # Add 1
        v6 = v2 * v5  # Multiply the two results
        return v6

# Initializing the model
model = CustomModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128

# Forward pass
output = model(input_tensor)

# Output the shape of the output tensor
print(output.shape)
