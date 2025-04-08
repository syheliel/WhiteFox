import torch

# Define the model
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: input channels = 3, output channels = 8, kernel size = 1
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise transposed convolution
        t1 = self.conv_transpose(x)
        # Add 3 to the output of the transposed convolution
        t2 = t1 + 3
        # Clamp the output of the addition operation to a minimum of 0
        t3 = torch.clamp_min(t2, 0)
        # Clamp the output of the previous operation to a maximum of 6
        t4 = torch.clamp_max(t3, 6)
        # Multiply the output of the transposed convolution by the output of the clamping operation
        t5 = t1 * t4
        # Divide the output of the multiplication operation by 6
        t6 = t5 / 6
        return t6

# Initialize the model
model = TransposeConvModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 input channels, 64x64 spatial dimensions

# Get the output from the model
output_tensor = model(input_tensor)

# Display the shape of the output tensor
print("Output shape:", output_tensor.shape)
