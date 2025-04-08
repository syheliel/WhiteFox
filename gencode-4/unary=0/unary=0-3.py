import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)                  # Apply pointwise convolution
        t2 = t1 * 0.5                      # Multiply by 0.5
        t3 = t1 * t1                       # Square the output
        t4 = t3 * t1                       # Cube the output
        t5 = t4 * 0.044715                 # Multiply by 0.044715
        t6 = t1 + t5                       # Add output of conv and t5
        t7 = t6 * 0.7978845608028654       # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)                # Apply hyperbolic tangent
        t9 = t8 + 1                        # Add 1 to the output of tanh
        t10 = t2 * t9                      # Multiply t2 by the output of tanh
        return t10

# Initializing the model
model = CustomModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass
output = model(input_tensor)

# Printing output shape
print(output.shape)  # Should match the output shape of the model
