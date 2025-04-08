import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Multiply
        return t6

# Initialize the model
model = CustomModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Get output from the model
output = model(input_tensor)

print("Output shape:", output.shape)
