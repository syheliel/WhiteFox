import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)  # Example input size of 128, output size of 64

    def forward(self, x1):
        l1 = self.linear(x1)  # Apply linear transformation
        l2 = l1 * 0.5  # Multiply by 0.5
        l3 = l1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        l4 = torch.erf(l3)  # Apply error function
        l5 = l4 + 1  # Add 1
        l6 = l2 * l5  # Final multiplication
        return l6

# Initializing the model
model = LinearModel()

# Input to the model
input_tensor = torch.randn(1, 128)  # Batch size of 1 and input size of 128
output = model(input_tensor)

# Display output shape
print(output.shape)
