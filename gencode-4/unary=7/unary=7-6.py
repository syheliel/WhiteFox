import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Pointwise linear transformation

    def forward(self, x):
        l1 = self.linear(x)          # Apply linear transformation
        l2 = l1 + 3                  # Add 3
        l3 = torch.clamp(l2, min=0)  # Clamp to a minimum of 0
        l4 = torch.clamp(l3, max=6)  # Clamp to a maximum of 6
        l5 = l1 * l4                 # Multiply by the output of the clamp operation
        l6 = l5 / 6                  # Divide by 6
        return l6

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(2, 10)  # Example input tensor with batch size of 2 and 10 features
output = model(input_tensor)

# Display output
print(output)

input_tensor = torch.randn(2, 10)  # Batch size of 2, 10 features
