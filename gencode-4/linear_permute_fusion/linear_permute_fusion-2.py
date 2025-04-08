import torch

# Model
class LinearPermuteModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        t1 = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)  # Apply linear transformation
        t2 = t1.permute(0, 2, 1)  # Permute the output tensor (swap last two dimensions)
        return t2

# Initializing the model with specific dimensions
input_dim = 4  # Input features
output_dim = 6  # Output features
model = LinearPermuteModel(input_dim, output_dim)

# Generating inputs to the model
# The input tensor must have at least 3 dimensions (batch size, features, sequence length)
input_tensor = torch.randn(2, 3, input_dim)  # Batch size of 2, 3 sequences, each with 4 features
output_tensor = model(input_tensor)

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output_tensor.shape)
