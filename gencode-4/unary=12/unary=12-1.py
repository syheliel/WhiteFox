import torch

# Model Definition
class GatedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)         # Apply pointwise convolution
        t2 = torch.sigmoid(t1)    # Apply sigmoid to the output of the convolution
        t3 = t1 * t2              # Multiply the output of the convolution by the output of the sigmoid function
        return t3

# Initializing the model
model = GatedModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)

# Display the shape of the output
print("Output shape:", output.shape)
