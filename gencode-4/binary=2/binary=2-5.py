import torch

class SubtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a pointwise convolution layer
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # No padding for kernel size 1

    def forward(self, x, other):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 - other    # Subtract 'other' from the output of the convolution
        return t2

# Initializing the model
model = SubtractionModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)
other_tensor = torch.randn(1, 8, 64, 64)   # 'other' tensor with the same shape as the output of the convolution

# Running the model
output = model(input_tensor, other_tensor)

# Print the output shape
print(output.shape)
