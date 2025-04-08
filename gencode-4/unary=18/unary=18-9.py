import torch

# Define the model
class SigmoidModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        # Apply convolution
        t1 = self.conv(x)
        # Apply sigmoid activation
        t2 = torch.sigmoid(t1)
        return t2

# Initialize the model
model = SigmoidModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

# Display the output shape
print(output.shape)
