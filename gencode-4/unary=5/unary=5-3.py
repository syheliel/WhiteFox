import torch

# Model Definition
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: input channels = 8, output channels = 3, kernel size = 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Multiply output by the result of the error function
        return t6

# Initializing the model
model = TransposedConvModel()

# Generating input tensor
# Input tensor shape: (batch_size, num_channels, height, width)
input_tensor = torch.randn(1, 8, 64, 64)  # Example input with 8 channels

# Forward pass through the model
output = model(input_tensor)

# Output shape
print(output.shape)
