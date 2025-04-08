import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolutional layer with weight and bias tensors
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.scalar = torch.tensor(1.5, dtype=torch.float32)  # A scalar for the binary operation

    def forward(self, x):
        t1 = self.conv(x)  # Apply convolution to the input tensor
        t2 = t1 + self.scalar  # Apply addition with the scalar
        return t2

# Initializing the model
model = Model()

# Generate input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

# Print the shape of the output tensor
print(output.shape)
