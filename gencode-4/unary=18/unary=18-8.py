import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x1):
        # Apply pointwise convolution
        t1 = self.conv(x1)
        # Apply sigmoid activation function
        t2 = torch.sigmoid(t1)
        return t2

# Initialize the model
model = Model()

# Generate input tensor (1 sample, 3 channels, 64x64 image)
input_tensor = torch.randn(1, 3, 64, 64)

# Forward pass through the model
output_tensor = model(input_tensor)

# Output the shape of the output tensor
print("Output shape:", output_tensor.shape)
