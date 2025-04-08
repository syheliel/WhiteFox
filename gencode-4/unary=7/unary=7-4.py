import torch

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Pointwise linear transformation

    def forward(self, x1):
        l1 = self.linear(x1)  # Apply linear transformation
        l2 = l1 + 3           # Add 3 to the output
        l3 = torch.clamp(l2, min=0)  # Clamp to min 0
        l4 = torch.clamp(l3, max=6)   # Clamp to max 6
        l5 = l1 * l4         # Multiply by the output of the clamp operation
        l6 = l5 / 6          # Divide by 6
        return l6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)  # Example input tensor with batch size 1 and 10 features
output = m(x1)

# Displaying the output
print(output)
