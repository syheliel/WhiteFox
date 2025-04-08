import torch

class SimpleReLUModel(torch.nn.Module):
    def __init__(self):
        super(SimpleReLUModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Apply ReLU activation function
        t2 = torch.nn.functional.relu(t1)
        return t2

# Initializing the model
model = SimpleReLUModel()

# Generating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

# Display the shapes of input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
