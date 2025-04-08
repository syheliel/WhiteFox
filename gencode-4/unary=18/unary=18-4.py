import torch

# Model definition
class SigmoidModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.sigmoid(t1)  # Apply sigmoid function
        return t2

# Initializing the model
model = SigmoidModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Forward pass through the model
output = model(input_tensor)

# Output the results
print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output.shape)
