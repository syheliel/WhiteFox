import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolutional layer with weights and bias
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Define a scalar attribute that will be used in the binary operation
        self.scalar = torch.tensor(2.0)  # Floating-point scalar

    def forward(self, x):
        # Apply convolution to the input tensor
        conv_output = self.conv(x)
        # Apply addition operation with the scalar
        result = conv_output + self.scalar
        return result

# Initializing the model
model = CustomModel()

# Creating an input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 64, 64)

# Getting the output from the model
output = model(input_tensor)
