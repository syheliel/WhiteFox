import torch

class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        l1 = self.linear(x)  # Apply pointwise linear transformation to the input tensor
        l2 = l1 * 0.5  # Multiply the output of the linear transformation by 0.5
        l3 = l1 * 0.7071067811865476  # Multiply the output of the linear transformation by 0.7071067811865476
        l4 = torch.erf(l3)  # Apply the error function to the output of the linear transformation
        l5 = l4 + 1  # Add 1 to the output of the error function
        l6 = l2 * l5  # Multiply the output of the linear transformation by the output of the error function
        return l6

# Initializing the model with input dimension of 10 and output dimension of 5
model = LinearModel(input_dim=10, output_dim=5)

# Generating input tensor for the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input feature size of 10

# Getting the output from the model
output_tensor = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output_tensor)
