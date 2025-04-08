import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution (kernel size 1)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    
    def forward(self, x1):
        t1 = self.conv(x1)          # Apply pointwise convolution
        t2 = t1 + 3                 # Add 3 to the output
        t3 = torch.clamp_min(t2, 0) # Clamp the output to a minimum of 0
        t4 = torch.clamp_max(t3, 6) # Clamp the output to a maximum of 6
        t5 = t4 / 6                 # Divide the output by 6
        return t5

# Initialize the model
m = Model()

# Create an input tensor
x1 = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Get the output from the model
output = m(x1)

# Display the output shape
print("Output shape:", output.shape)
