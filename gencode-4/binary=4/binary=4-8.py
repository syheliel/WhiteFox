import torch

# Model
class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.other = torch.randn(1, output_dim)  # This tensor will be added to the output

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 + self.other  # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
input_dim = 10
output_dim = 5
model = LinearModel(input_dim, output_dim)

# Input to the model
x_input = torch.randn(1, input_dim)  # Batch size of 1 and input dimension
output = model(x_input)

# Print the output
print(output)
