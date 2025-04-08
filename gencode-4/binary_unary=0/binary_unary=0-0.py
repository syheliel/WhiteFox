import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution layer with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, other):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 + other    # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)  # Apply ReLU activation
        return t3

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
other_tensor = torch.randn(1, 8, 64, 64)   # Must match the output shape of the convolution (1, 8, 64, 64)

# Forward pass through the model
output = model(input_tensor, other_tensor)

# Print output shape
print(output.shape)
