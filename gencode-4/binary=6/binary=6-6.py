import torch

# Model Definition
class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.other = torch.tensor(1.0)  # This can be any tensor or scalar to subtract

    def forward(self, x):
        t1 = self.linear(x)           # Apply a linear transformation
        t2 = t1 - self.other          # Subtract 'other' from the output
        return t2

# Initializing the model
input_dim = 10  # Example input dimensionality
output_dim = 5  # Example output dimensionality
model = LinearModel(input_dim, output_dim)

# Inputs to the model
input_tensor = torch.randn(1, input_dim)  # Batch size of 1, with input_dim features
output = model(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
