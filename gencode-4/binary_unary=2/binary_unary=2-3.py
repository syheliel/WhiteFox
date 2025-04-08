import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        # Define a scalar to subtract
        self.other = torch.tensor(0.5)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Subtract 'other' from the output of the convolution
        t2 = t1 - self.other
        # Apply ReLU activation function
        t3 = torch.relu(t2)
        return t3

# Initialize the model
model = CustomModel()

# Create an input tensor (batch size, channels, height, width)
input_tensor = torch.randn(1, 3, 64, 64)

# Get the model output
output = model(input_tensor)

# Print the output shape
print(output.shape)
