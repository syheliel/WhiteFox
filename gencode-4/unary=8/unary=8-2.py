import torch

# Model definition
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution to the input tensor
        t2 = t1 + 3  # Add 3 to the output of the transposed convolution
        t3 = torch.clamp_min(t2, 0)  # Clamp the output of the addition operation to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output of the previous operation to a maximum of 6
        t5 = t1 * t4  # Multiply the output of the transposed convolution by the output of the clamping operation
        t6 = t5 / 6  # Divide the output of the multiplication operation by 6
        return t6

# Initializing the model
model = TransposeConvModel()

# Generating input tensor
input_tensor = torch.randn(1, 16, 64, 64)  # Batch size of 1, 16 input channels, 64x64 spatial dimensions

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)  # Should reflect the output shape after the transposed convolution
