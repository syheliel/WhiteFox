import torch

class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x, other):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 + other      # Add another tensor to the output of the linear transformation
        t3 = torch.nn.functional.relu(t2)  # Apply the ReLU activation function to the result
        return t3

# Initialize the model with specific input and output sizes
input_size = 10
output_size = 5
model = LinearModel(input_size, output_size)

# Generate input tensor
x = torch.randn(1, input_size)  # A batch size of 1 and input size of 10
other = torch.randn(1, output_size)  # Another tensor to be added, matching the output size

# Get the output from the model
output = model(x, other)

# Print the output
print(output)
