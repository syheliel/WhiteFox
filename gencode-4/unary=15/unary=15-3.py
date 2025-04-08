import torch

# Model definition
class SimpleReLUModel(torch.nn.Module):
    def __init__(self):
        super(SimpleReLUModel, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution with kernel size 1 to the input tensor
        t2 = torch.relu(t1)  # Apply the ReLU activation function
        return t2

# Initializing the model
model = SimpleReLUModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 color channels, 64x64 image
output = model(input_tensor)

# Output shape
print("Output shape:", output.shape)
