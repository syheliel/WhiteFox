import torch

# Model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Pointwise linear transformation from 10 to 5 features

    def forward(self, x):
        l1 = self.linear(x)  # Apply linear transformation
        l2 = l1 * 0.5  # Multiply by 0.5
        l3 = l1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        l4 = torch.erf(l3)  # Apply error function
        l5 = l4 + 1  # Add 1 to the output of the error function
        l6 = l2 * l5  # Multiply the transformed output with the adjusted error function
        return l6

# Initializing the model
model = LinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = model(input_tensor)

# Display the output
print(output)
