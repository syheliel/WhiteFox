import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)  # Pointwise linear transformation

    def forward(self, x):
        l1 = self.linear(x)  # Apply linear transformation
        l2 = l1 + 3          # Add 3 to the output
        l3 = torch.clamp(l2, min=0)  # Clamp the output to a minimum of 0
        l4 = torch.clamp(l3, max=6)   # Clamp the output to a maximum of 6
        l5 = l1 * l4         # Multiply by the output of the clamp operation
        l6 = l5 / 6          # Divide by 6
        return l6

# Initializing the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Input tensor with batch size 1 and feature size 10

# Forward pass through the model
output = model(input_tensor)

# Display output
print("Output:", output)
