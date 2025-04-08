import torch

# Define the model class
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, other):
        t1 = self.conv(x1)  # Apply pointwise convolution
        t2 = t1 + other     # Add another tensor to the output of the convolution
        t3 = torch.nn.functional.relu(t2)  # Apply ReLU activation
        return t3

# Initializing the model
model = Model()

# Generating inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Input tensor with shape (batch_size, channels, height, width)
other = torch.randn(1, 8, 64, 64)  # Another tensor with the same spatial dimensions

# Forward pass
output = model(x1, other)

# Display the output shape
print(output.shape)
