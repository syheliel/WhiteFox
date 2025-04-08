import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
 
    def forward(self, x1):
        l1 = self.linear(x1)  # Apply pointwise linear transformation
        l2 = l1 * 0.5         # Multiply the output of the linear transformation by 0.5
        l3 = l1 * 0.7071067811865476  # Multiply the output of the linear transformation by 0.7071067811865476
        l4 = torch.erf(l3)   # Apply the error function to the output of the linear transformation
        l5 = l4 + 1          # Add 1 to the output of the error function
        l6 = l2 * l5         # Multiply the output of the linear transformation by the output of the error function
        return l6

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Inputs to the model
x1 = torch.randn(1, input_dim)  # Batch size of 1 and input dimension
output = model(x1)

# Output from the model
print(output)
