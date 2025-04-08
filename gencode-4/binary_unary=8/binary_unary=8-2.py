import torch

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1
        self.other = torch.randn(1, 8, 64, 64)  # Another tensor to be added

    def forward(self, x1):
        t1 = self.conv(x1)           # Apply pointwise convolution to the input tensor
        t2 = t1 + self.other         # Add another tensor to the output of the convolution
        t3 = torch.relu(t2)          # Apply ReLU activation function to the result
        return t3

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Output shape
print(output.shape)  # Should be (1, 8, 64, 64)
