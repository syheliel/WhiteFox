import torch

# Define the model
class TransposeConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)

    def forward(self, input_tensor, negative_slope=0.01):
        t1 = self.conv_transpose(input_tensor)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create a mask where output is greater than 0
        t3 = t1 * negative_slope  # Multiply by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask to the output
        return t4

# Initialize the model
model = TransposeConvModel()

# Generate an input tensor
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Get the output from the model
output = model(input_tensor)

# Print the shape of the output
print("Output shape:", output.shape)
