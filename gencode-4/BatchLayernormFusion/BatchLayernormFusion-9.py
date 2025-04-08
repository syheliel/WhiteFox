import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Multiply by 0.5
        t2 = t1 * 0.5
        # Multiply by 0.7071067811865476
        t3 = t1 * 0.7071067811865476
        # Apply the error function
        t4 = torch.erf(t3)
        # Add 1
        t5 = t4 + 1
        # Multiply the outputs
        t6 = t2 * t5
        return t6

# Initializing the model
model = CustomModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
# Forward pass through the model
output_tensor = model(input_tensor)

# Example output
print(output_tensor.shape)  # Should be (1, 10, 128, 128)
