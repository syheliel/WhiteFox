import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1)  # Pointwise convolution

    def forward(self, x1, other_tensor):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + other_tensor  # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initialize the model
model = Model()

# Generate input tensors
x1 = torch.randn(1, 3, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)
other_tensor = torch.randn(1, 8, 64, 64)  # Another tensor to add, must match the output shape of the conv layer

# Forward pass through the model
output = model(x1, other_tensor)

# Output shape
print(output.shape)  # Should be (1, 8, 64, 64)
