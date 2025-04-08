import torch

class PermuteLinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # A linear layer with a specified input and output dimension
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Permute the input tensor by swapping the last two dimensions
        t1 = x.permute(0, 2, 1)  # Assuming input tensor has shape [batch_size, seq_length, features]
        # Apply linear transformation to the permuted tensor
        t2 = self.linear(t1)
        return t2

# Create an instance of the model
input_dim = 128  # Number of features
output_dim = 64  # Number of output features
model = PermuteLinearModel(input_dim, output_dim)

# Generate a random input tensor with shape [batch_size, seq_length, features]
# Here we assume batch_size = 1 and seq_length = 10 for demonstration
x_input = torch.randn(1, 10, input_dim)

# Forward pass through the model
output = model(x_input)

print(output.shape)  # Should output: torch.Size([1, 10, 64])
