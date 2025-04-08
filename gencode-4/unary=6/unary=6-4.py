import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 + 3        # Add 3 to the output of the convolution
        t3 = torch.clamp_min(t2, 0)  # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6)   # Clamp the output to a maximum of 6
        t5 = t1 * t4      # Multiply the output of the convolution by the output of the clamp operation
        t6 = t5 / 6       # Divide the result by 6
        return t6

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Print the output shape
print(output.shape)
