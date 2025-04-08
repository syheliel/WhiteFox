import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        # A scalar to subtract from the convolution output
        self.other = 0.1

    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Subtract a scalar from the convolution output
        t2 = t1 - self.other
        # Apply ReLU activation
        t3 = torch.relu(t2)
        return t3

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size
output = model(input_tensor)

print(output.shape)  # Output shape
