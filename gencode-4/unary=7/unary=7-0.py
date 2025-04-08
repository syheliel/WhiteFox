import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Input size 10, output size 5

    def forward(self, x):
        l1 = self.linear(x)  # Apply pointwise linear transformation
        l2 = l1 + 3          # Add 3 to the output of the linear transformation
        l3 = torch.clamp(l2, min=0)  # Clamp the output of the addition operation to a minimum of 0
        l4 = torch.clamp(l3, max=6)   # Clamp the output of the previous clamp operation to a maximum of 6
        l5 = l1 * l4          # Multiply the output of the linear transformation by the output of the clamp operation
        l6 = l5 / 6           # Divide the output of the multiplication operation by 6
        return l6

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input dimension of 10
output = model(input_tensor)

# Display the output
print(output)
