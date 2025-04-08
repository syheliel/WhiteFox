import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)  # Changing number of output channels to 16

    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 - other  # Subtract the other tensor from the convolution output
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
model = CustomModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
other_tensor = torch.randn(1, 16, 64, 64)  # Same shape as conv output

# Forward pass
output = model(input_tensor, other_tensor)

print(output.shape)  # Checking the output shape
