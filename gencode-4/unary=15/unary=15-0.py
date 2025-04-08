import torch

# Model definition
class SimpleReLUModel(torch.nn.Module):
    def __init__(self):
        super(SimpleReLUModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.relu(t1)  # Apply ReLU activation
        return t2

# Initializing the model
model = SimpleReLUModel()

# Creating an input tensor with a batch size of 1, 3 channels, and 64x64 spatial dimensions
input_tensor = torch.randn(1, 3, 64, 64)

# Running the model on the input tensor
output_tensor = model(input_tensor)

# Print the output tensor shape
print(output_tensor.shape)
