import torch

class LeakyReLUModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.negative_slope = negative_slope

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 > 0  # Create a boolean tensor
        t3 = t1 * self.negative_slope  # Multiply by the negative slope
        t4 = torch.where(t2, t1, t3)  # Apply the Leaky ReLU condition
        return t4

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
negative_slope = 0.01  # Example negative slope
model = LeakyReLUModel(input_dim, output_dim, negative_slope)

# Inputs to the model
x_input = torch.randn(1, input_dim)  # Batch size of 1, input dimension of 10

# Forward pass
output = model(x_input)

print("Input Tensor:")
print(x_input)
print("\nOutput Tensor:")
print(output)
