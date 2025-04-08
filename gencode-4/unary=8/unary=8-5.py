import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 + 3  # Add 3 to the output of the transposed convolution
        t3 = torch.clamp_min(t2, 0)  # Clamp the output of the addition operation to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output of the previous operation to a maximum of 6
        t5 = t1 * t4  # Multiply the output of the transposed convolution by the output of the clamping operation
        t6 = t5 / 6  # Divide the output of the multiplication operation by 6
        return t6

# Initializing the model
model = TransposeConvModel()

# Generating input tensor for the model
# The input shape must match the expected input channels of the transposed convolution.
# Since we've defined the transposed convolution to take 8 input channels, we create an input tensor accordingly.
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Passing the input through the model
output_tensor = model(input_tensor)

# Output tensor shape
print("Output tensor shape:", output_tensor.shape)
