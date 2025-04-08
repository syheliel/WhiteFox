import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=0)  # Pointwise transposed convolution
        self.negative_slope = 0.2  # Define the negative slope for Leaky ReLU

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create mask where output is greater than 0
        t3 = t1 * self.negative_slope  # Multiply output by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply mask
        return t4

# Initializing the model
model = Model()

# Input to the model
input_tensor = torch.randn(1, 8, 64, 64)  # Example input tensor with batch size 1, 8 channels, 64x64 spatial dimensions

# Getting the output from the model
output = model(input_tensor)

# Print the output shape
print(f'Output shape: {output.shape}')
