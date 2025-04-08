import torch

# Model definition
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLUModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)  # Pointwise convolution
        self.negative_slope = negative_slope  # Negative slope for Leaky ReLU

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 > 0  # Create boolean mask
        t3 = t1 * self.negative_slope  # Multiply by negative slope
        t4 = torch.where(t2, t1, t3)  # Apply torch.where
        return t4

# Initializing the model
model = LeakyReLUModel(negative_slope=0.01)

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output_tensor = model(input_tensor)

# Output tensor
print(output_tensor)
