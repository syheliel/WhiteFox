import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)          # Apply pointwise convolution
        t2 = t1 + 3                # Add 3 to the output of the convolution
        t3 = torch.clamp_min(t2, 0) # Clamp the output of the addition operation to a minimum of 0
        t4 = torch.clamp_max(t3, 6) # Clamp the output of the previous operation to a maximum of 6
        t5 = t4 / 6                # Divide the output of the previous operation by 6
        return t5

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output_tensor = model(input_tensor)

# Display the shape of the output tensor
print(output_tensor.shape)
