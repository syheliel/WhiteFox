import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.other = 0.25  # Scalar value to subtract

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 - self.other  # Subtract 'other' from the output of the convolution
        return t2

# Initializing the model
model = Model()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, and 64x64 image size

# Get the output from the model
output = model(input_tensor)

print("Output shape:", output.shape)
