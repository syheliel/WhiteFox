import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolutional layer with 3 input channels, 8 output channels, and kernel size 3
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Define a scalar to be used in the binary operation
        self.scalar_add = torch.tensor(0.5, dtype=torch.float32)

    def forward(self, x1):
        # Apply convolution
        t1 = self.conv(x1)
        # Apply binary operation (addition) with the scalar
        t2 = t1 + self.scalar_add
        return t2

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Get the output from the model
output = model(input_tensor)

# Display the output shape
print(output.shape)

input_tensor = torch.randn(1, 3, 64, 64)
