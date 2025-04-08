import torch

class SplitStackTanhModel(torch.nn.Module):
    def __init__(self, split_size):
        super().__init__()
        self.split_size = split_size

    def forward(self, x):
        # Split the input tensor into chunks along the channel dimension
        t1 = torch.split(x, self.split_size, dim=1)  # Assuming dim=1 (channels)
        # Stack the chunks along a new dimension (dim=0)
        t2 = torch.stack(t1, dim=0)
        # Apply the hyperbolic tangent function to the output of the stack operation
        t3 = torch.tanh(t2)
        return t3

# Initializing the model with a split size of 2
split_size = 2
model = SplitStackTanhModel(split_size)

# Creating an input tensor with the shape (batch_size, channels, height, width)
x_input = torch.randn(1, 4, 64, 64)  # Example input tensor: batch size 1, 4 channels, 64x64 image

# Getting the output from the model
output = model(x_input)

print("Output shape:", output.shape)
