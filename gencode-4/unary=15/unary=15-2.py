import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.relu(t1)  # Apply ReLU activation function
        return t2

# Initializing the model
model = SimpleModel()

# Generating inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image size

# Forward pass to get the output
output = model(input_tensor)

# Display the output shape
print(output.shape)
