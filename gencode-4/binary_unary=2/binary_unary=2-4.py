import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)                 # Apply pointwise convolution
        t2 = t1 - 0.5                     # Subtract scalar 0.5 from the output
        t3 = torch.nn.functional.relu(t2) # Apply ReLU activation function
        return t3

# Initializing the model
model = CustomModel()

# Input tensor to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass
output = model(input_tensor)

# Printing the output shape
print(output.shape)
