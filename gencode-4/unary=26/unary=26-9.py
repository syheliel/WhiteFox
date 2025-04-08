import torch

# Define the model
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)  # Input channels: 8, Output channels: 3

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create a mask where the output of the convolution is greater than 0
        negative_slope = 0.01  # Define a negative slope for the Leaky ReLU
        t3 = t1 * negative_slope  # Multiply the output of the convolution by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask to the output of the convolution
        return t4

# Initializing the model
model = ModelTransposeConv()

# Generate an input tensor
# Since the input channels of the transposed convolution are set to 8, we create an input tensor accordingly
x1 = torch.randn(1, 8, 64, 64)  # Batch size: 1, Channels: 8, Height: 64, Width: 64

# Forward pass through the model
output = model(x1)

# Print the shape of the output tensor
print(output.shape)
