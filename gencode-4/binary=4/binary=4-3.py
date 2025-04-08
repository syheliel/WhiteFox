import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.other = torch.randn(output_dim)  # This will be the tensor added to the output

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + self.other  # Add another tensor to the output
        return t2

# Initializing the model with specific dimensions
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Creating an input tensor
x1 = torch.randn(1, input_dim)  # Batch size of 1

# Getting the output from the model
output = model(x1)

# Print the output
print(output)
