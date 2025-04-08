import torch

class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, other):
        t1 = self.linear(x)            # Apply a linear transformation to the input tensor
        t2 = t1 + other                # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)            # Apply the ReLU activation function to the result
        return t3

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Generating input tensor
x = torch.randn(1, input_dim)      # Batch size of 1 and input dimension of 10
other = torch.randn(1, output_dim)  # Another tensor to add to the output of the linear transformation

# Forward pass
output = model(x, other)

print("Input tensor (x):", x)
print("Other tensor:", other)
print("Output tensor:", output)
