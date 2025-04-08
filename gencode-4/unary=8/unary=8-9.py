import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 + 3                   # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)  # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)   # Clamp the output to a maximum of 6
        t5 = t1 * t4                  # Multiply the transposed convolution output by the clamped value
        t6 = t5 / 6                   # Divide the result by 6
        return t6

# Initializing the model
model = TransposeConvModel()

# Generating input tensor
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Output shape
print(output.shape)
