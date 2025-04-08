import torch

# Model
class SimpleLinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x, other):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + other      # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
input_size = 10   # Example input size
output_size = 5   # Example output size
model = SimpleLinearModel(input_size, output_size)

# Inputs to the model
x_input = torch.randn(1, input_size)  # Input tensor of shape (1, input_size)
other_tensor = torch.randn(1, output_size)  # Another tensor of shape (1, output_size)

# Output of the model
output = model(x_input, other_tensor)

# Print the output
print(output)
