import torch

class NewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)  # Pointwise convolution with kernel size 1

    def forward(self, x):
        t1 = self.conv(x)  # Apply pointwise convolution
        t2 = t1 - 0.3      # Subtract a scalar (0.3 in this case)
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initializing the model
model = NewModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output_tensor = model(input_tensor)

print("Output shape:", output_tensor.shape)
