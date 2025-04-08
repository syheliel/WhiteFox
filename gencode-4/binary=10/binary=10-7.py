import torch

# Model Definition
class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x, other):
        t1 = self.linear(x)      # Apply a linear transformation
        t2 = t1 + other          # Add another tensor to the output
        return t2

# Initializing the model
input_size = 10   # Example input size
output_size = 5   # Example output size
model = LinearModel(input_size, output_size)

# Generating input tensor
x = torch.randn(1, input_size)  # Batch size of 1 and input size of 10
other = torch.randn(1, output_size)  # Another tensor to add, with size matching the output size

# Forward pass through the model
output = model(x, other)

# Printing the output
print(output)
