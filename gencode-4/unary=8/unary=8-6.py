import torch

# Model definition
class TransposedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise transposed convolution: input channels = 8, output channels = 3, kernel size = 1
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        t1 = self.conv_transpose(x)  # Apply pointwise transposed convolution
        t2 = t1 + 3                   # Add 3 to the output
        t3 = torch.clamp_min(t2, 0)  # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output to a maximum of 6
        t5 = t1 * t4                  # Multiply the output of the transposed convolution by the clamped output
        t6 = t5 / 6                   # Divide the output by 6
        return t6

# Initializing the model
model = TransposedConvModel()

# Creating an input tensor for the model
# Batch size = 1, Input channels = 8, Height = 64, Width = 64
input_tensor = torch.randn(1, 8, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Output shape
print(output.shape)
