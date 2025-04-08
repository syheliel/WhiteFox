import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        l1 = self.linear(x)  # Pointwise linear transformation
        l2 = l1 * 0.5  # Multiply by 0.5
        l3 = l1 * 0.7071067811865476  # Multiply by 0.7071067811865476
        l4 = torch.erf(l3)  # Apply the error function
        l5 = l4 + 1  # Add 1
        l6 = l2 * l5  # Multiply by the output of the error function
        return l6

# Initializing the model with specified input and output dimensions
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Generating a random input tensor for the model
x = torch.randn(1, input_dim)  # Batch size of 1

# Getting the output of the model
output = model(x)

# Print output shape for verification
print("Output shape:", output.shape)
