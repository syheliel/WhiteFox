import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.negative_slope = 0.1  # Example negative slope for Leaky ReLU

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create mask where output is greater than 0
        t3 = t1 * self.negative_slope  # Multiply output by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the mask to the output
        return t4

# Initializing the model
model = Model()

# Creating an input tensor
input_tensor = torch.randn(1, 8, 64, 64)  # Batch size of 1, 8 channels, 64x64 spatial dimensions

# Forward pass through the model
output = model(input_tensor)

# Output shape
print(output.shape)  # Should be [1, 3, 64, 64] since we have 3 output channels
