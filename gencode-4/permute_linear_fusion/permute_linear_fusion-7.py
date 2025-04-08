import torch

# Define the model
class CustomModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Permute the tensor to swap the last two dimensions
        t1 = x.permute(0, 2, 1)  # Assuming x has shape (batch_size, seq_length, features)
        # Apply linear transformation
        t2 = self.linear(t1)
        return t2

# Initialize the model
input_dim = 64  # Number of features
output_dim = 32  # Number of output features
model = CustomModel(input_dim, output_dim)

# Create an input tensor of shape (batch_size, seq_length, features)
# For example, a batch of 10 sequences, each with 5 time steps and 64 features
input_tensor = torch.randn(10, 5, input_dim)

# Forward pass through the model
output_tensor = model(input_tensor)

# Print the output shape
print(output_tensor.shape)  # Expected shape: (10, 5, output_dim)
