import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, x):
        t1 = self.conv(x)  # Apply convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by constant
        t4 = torch.erf(t3)  # Apply error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Final multiplication
        return t6

# Initialize the model
model = CustomModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image

# Get output from the model
output_tensor = model(input_tensor)
