import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x, other):
        t1 = self.linear(x)          # Apply a linear transformation to the input tensor
        t2 = t1 + other              # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)          # Apply the ReLU activation function to the result
        return t3

# Initialize the model with input and output dimensions
input_dim = 10
output_dim = 5
model = Model(input_dim, output_dim)

# Generate input tensor and the 'other' tensor
x_tensor = torch.randn(1, input_dim)  # Input tensor with shape (1, input_dim)
other_tensor = torch.randn(1, output_dim)  # Other tensor with shape (1, output_dim)

# Get the output by passing the input tensor and the 'other' tensor to the model
output = model(x_tensor, other_tensor)

# Display the output
print(output)
