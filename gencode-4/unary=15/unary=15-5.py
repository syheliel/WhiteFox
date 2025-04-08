import torch

# Define the model
class SimpleReLUModel(torch.nn.Module):
    def __init__(self):
        super(SimpleReLUModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)  # Apply pointwise convolution
        # Apply the ReLU activation function
        t2 = torch.relu(t1)  # Apply ReLU
        return t2

# Initialize the model
model = SimpleReLUModel()

# Create an input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output shape
print(output_tensor.shape)  # Should be (1, 8, 64, 64)
