import torch

# Model
class NewModel(torch.nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)  # Pointwise convolution

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        t4 = torch.erf(t3)  # Apply the error function
        t5 = t4 + 1  # Add 1
        t6 = t2 * t5  # Multiply t2 by t5
        return t6

# Initializing the model
model = NewModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = model(input_tensor)
