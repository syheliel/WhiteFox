import torch

class AlternativeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)  # Changed number of output channels

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
model = AlternativeModel()

# Generating input tensor
input_tensor = torch.randn(1, 3, 128, 128)  # Changed dimensions of the input tensor
output = model(input_tensor)

# Output shape for verification
print(output.shape)
