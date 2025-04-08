import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x1, other):
        t1 = self.conv(x1)          # Apply pointwise convolution
        t2 = t1 + other             # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)        # Apply the ReLU activation function
        return t3

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)  # Input tensor with batch size 1, 3 color channels, and 64x64 spatial dimensions
other = torch.randn(1, 8, 64, 64)  # Another tensor to be added (must match the output shape of the convolution)

# Forward pass
output = model(x1, other)
