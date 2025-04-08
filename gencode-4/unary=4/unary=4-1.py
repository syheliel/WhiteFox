import torch

# Model Definition
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Pointwise linear transformation

    def forward(self, x):
        l1 = self.linear(x)  # Apply linear transformation
        l2 = l1 * 0.5  # Multiply by 0.5
        l3 = l1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        l4 = torch.erf(l3)  # Apply the error function
        l5 = l4 + 1  # Add 1
        l6 = l2 * l5  # Multiply the outputs
        return l6

# Initialize the model
model = LinearModel()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10

# Forward pass through the model
output = model(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
