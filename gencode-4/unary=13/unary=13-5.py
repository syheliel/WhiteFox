import torch

# Model definition
class GatedLinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        t1 = self.linear(x)            # Apply a linear transformation to the input tensor
        t2 = torch.sigmoid(t1)        # Apply the sigmoid function to the output of the linear transformation
        t3 = t1 * t2                  # Multiply the output of the linear transformation by the output of the sigmoid function
        return t3

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = GatedLinearModel(input_dim, output_dim)

# Inputs to the model
x_input = torch.randn(1, input_dim)  # Creating a random input tensor with shape (1, input_dim)
output = model(x_input)

print("Input Tensor:", x_input)
print("Output Tensor:", output)
