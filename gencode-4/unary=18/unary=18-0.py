import torch

class SigmoidConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = torch.sigmoid(t1)  # Apply sigmoid activation
        return t2

# Initialize the model
model = SigmoidConvModel()

# Generate input tensor
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image

# Get the model output
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
