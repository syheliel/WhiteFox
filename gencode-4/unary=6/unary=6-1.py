import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)            # Apply pointwise convolution
        t2 = t1 + 3                  # Add 3 to the output of the convolution
        t3 = torch.clamp_min(t2, 0)  # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)  # Clamp the output to a maximum of 6
        t5 = t1 * t4                 # Multiply the output of the convolution by the output of the clamp operation
        t6 = t5 / 6                  # Divide the output by 6
        return t6

# Initializing the model
model = CustomModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, height and width of 64

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output shape
print(output_tensor.shape)
