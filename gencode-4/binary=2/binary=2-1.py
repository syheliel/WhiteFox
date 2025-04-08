import torch

class SubtractModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise convolutional layer
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Kernel size 1, no padding

    def forward(self, x):
        # Apply the convolution
        t1 = self.conv(x)
        # Define a scalar value to subtract from the convolution output
        other = 0.25
        # Subtract 'other' from the output of the convolution
        t2 = t1 - other
        return t2

# Initializing the model
model = SubtractModel()

# Generating input tensor for the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size = 1, Channels = 3, Height = 64, Width = 64

# Forward pass through the model
output_tensor = model(input_tensor)

# Displaying the output shape
print(output_tensor.shape)
