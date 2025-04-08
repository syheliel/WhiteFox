import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise transposed convolution layer
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv_transpose(x1)  # Apply pointwise transposed convolution
        t2 = t1 + 3  # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)  # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output to a maximum of 6
        t5 = t1 * t4  # Multiply the output of the transposed convolution by the clamped output
        t6 = t5 / 6  # Divide the result by 6
        return t6

# Initializing the model
model = Model()

# Generate a random input tensor
# Assuming input to the transposed convolution layer has 8 channels, height and width of 64
x1 = torch.randn(1, 8, 64, 64)

# Forward pass through the model
output = model(x1)

# Output shape
print(output.shape)
