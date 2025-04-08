import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a convolution layer with 3 input channels and 8 output channels, kernel size 3
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Define an attribute tensor to be used in the binary operation
        self.other_tensor = torch.tensor(0.5, dtype=torch.float32)  # This is a scalar

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply convolution to the input tensor
        t2 = t1 + self.other_tensor     # Apply addition with the output of the convolution and the other tensor
        return t2

# Initializing the model
model = MyModel()

# Generate an input tensor with the shape [batch_size, channels, height, width]
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output = model(input_tensor)

print(output.shape)  # Output shape should be [1, 8, 64, 64]
