import torch

# Model Definition
class SimpleReLUModel(torch.nn.Module):
    def __init__(self):
        super(SimpleReLUModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Apply pointwise convolution
        t1 = self.conv(x)
        # Apply ReLU activation
        t2 = torch.nn.functional.relu(t1)
        return t2

# Initializing the model
model = SimpleReLUModel()

# Creating an input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 spatial dimensions

# Forward pass through the model
output_tensor = model(input_tensor)

# Output the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)
