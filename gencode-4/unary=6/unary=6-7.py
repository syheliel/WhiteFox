import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        t1 = self.conv(x1)            # Apply pointwise convolution
        t2 = t1 + 3                   # Add 3 to the output of the convolution
        t3 = torch.clamp_min(t2, 0)   # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)   # Clamp the output to a maximum of 6
        t5 = t1 * t4                  # Multiply the output of convolution by the clamped output
        t6 = t5 / 6                   # Divide the result by 6
        return t6

# Initializing the model
model = Model()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

# Display the output shape
print("Output shape:", output.shape)
