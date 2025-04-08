import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071...
        t4 = torch.erf(t3)  # Apply error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Final computation
        return t6

# Initializing the model
model = CustomModel()

# Generating input tensor for the model
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 color channels, 128x128 image size

# Forward pass through the model
output = model(input_tensor)
print(output.shape)  # Print the shape of the output tensor
