import torch

# Model Definition
class ModelTransposeConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 + 3  # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)  # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output to a maximum of 6
        t5 = t1 * t4  # Multiply the output of the transposed convolution by the clamped output
        t6 = t5 / 6  # Divide the output by 6
        return t6

# Initializing the model
model = ModelTransposeConv()

# Creating an input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

# Printing the output shape
print("Output shape:", output_tensor.shape)
