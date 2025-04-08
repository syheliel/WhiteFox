import torch

# Model Definition
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution with kernel size 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 > 0  # Create mask where output is greater than 0
        negative_slope = 0.01  # Define a negative slope
        t3 = t1 * negative_slope  # Multiply output by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply mask to output
        return t4

# Initializing the model
model = TransposedConvModel()

# Inputs to the model
input_tensor = torch.randn(1, 8, 64, 64)  # Example input tensor with shape (batch_size, channels, height, width)

# Forward pass through the model
output_tensor = model(input_tensor)
