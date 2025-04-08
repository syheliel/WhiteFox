import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Input dimension of 10 and output dimension of 5

    def forward(self, x):
        l1 = self.linear(x)  # Apply linear transformation
        l2 = l1 + 3  # Add 3
        l3 = torch.clamp(l2, min=0)  # Clamp to minimum of 0
        l4 = torch.clamp(l3, max=6)  # Clamp to maximum of 6
        l5 = l1 * l4  # Multiply by the output of the clamp operation
        l6 = l5 / 6  # Divide by 6
        return l6

# Initializing the model
model = Model()

# Creating an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, input dimension of 10

# Forward pass through the model
output = model(input_tensor)

# Displaying the output
print(output)
