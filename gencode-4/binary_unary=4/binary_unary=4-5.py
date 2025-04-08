import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, other):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + other      # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = SimpleModel(input_dim, output_dim)

# Inputs to the model
x = torch.randn(1, input_dim)  # Batch size of 1
other = torch.randn(1, output_dim)  # An additional tensor to add

# Forward pass
output = model(x, other)

# Print the output
print(output)
