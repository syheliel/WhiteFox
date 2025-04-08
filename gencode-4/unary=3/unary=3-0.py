import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply the output by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply the output by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1 to the output of the error function
        t6 = t2 * t5  # Multiply the output by the modified error function
        return t6

# Initializing the model
model = CustomModel()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Forward pass through the model
output = model(input_tensor)

# Displaying the output shape
print("Output shape:", output.shape)
