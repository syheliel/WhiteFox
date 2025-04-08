import torch

# Model definition
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.other = 0.5  # Scalar to subtract from the output

    def forward(self, input_tensor):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 - self.other           # Subtract the scalar 'other'
        t3 = torch.relu(t2)           # Apply ReLU activation function
        return t3

# Initializing the model
model = CustomModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Output the shape of the output tensor
print(output.shape)
