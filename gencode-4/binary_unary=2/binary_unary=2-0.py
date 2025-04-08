import torch

# Model Definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 - other  # Subtract 'other' from the convolution output
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initialize the model
model = CustomModel()

# Generate an input tensor (batch size of 1, 3 channels, 64x64)
input_tensor = torch.randn(1, 3, 64, 64)

# Generate 'other' tensor to subtract from the convolution output
# It should have the same shape as the output of the convolution (batch size, number of output channels, height, width)
other = torch.randn(1, 8, 64, 64)

# Get the model output
output = model(input_tensor, other)

# Print the output shape
print(output.shape)  # Should be (1, 8, 64, 64)
