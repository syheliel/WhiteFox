import torch

# Define the model class
class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.other = torch.randn(1, output_dim)  # Create a tensor to be added

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + self.other  # Add another tensor to the output of the linear transformation
        return t2

# Initialize the model with specific input and output dimensions
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = LinearModel(input_dim, output_dim)

# Create an input tensor for the model
input_tensor = torch.randn(1, input_dim)  # Batch size of 1 and input dimension of 10

# Forward pass through the model
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)
