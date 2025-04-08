import torch

# Model definition
class SubtractionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, other):
        t1 = self.conv(input_tensor)  # Apply pointwise convolution
        t2 = t1 - other  # Subtract 'other' from the output of the convolution
        return t2

# Initializing the model
model = SubtractionModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)  # Example input tensor
other = torch.tensor(0.5).expand_as(model.conv(input_tensor))  # Scalar 'other' expanded to match output shape

# Forward pass
output = model(input_tensor, other)

# Output shape
print("Output shape:", output.shape)
