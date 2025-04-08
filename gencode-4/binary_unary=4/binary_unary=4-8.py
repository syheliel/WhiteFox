import torch

class CustomModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, input_tensor, other):
        t1 = self.linear(input_tensor)  # Apply linear transformation
        t2 = t1 + other  # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
input_dimension = 10  # Define input dimension
output_dimension = 5  # Define output dimension
model = CustomModel(input_dimension, output_dimension)

# Inputs to the model
input_tensor = torch.randn(1, input_dimension)  # Batch size of 1
other_tensor = torch.randn(1, output_dimension)  # Must match the output dimension of the linear layer

# Forward pass
output = model(input_tensor, other_tensor)

# Print the output
print(output)
