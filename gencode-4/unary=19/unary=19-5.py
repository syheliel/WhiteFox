import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        t1 = self.linear(x)          # Apply a linear transformation
        t2 = torch.sigmoid(t1)      # Apply the sigmoid function
        return t2

# Initialize the model
input_size = 10  # Example input size
output_size = 1  # Output size for binary classification
model = Model(input_size, output_size)

# Generate an input tensor
x_input = torch.randn(1, input_size)  # Batch size of 1 and input size of 10

# Get the output from the model
output = model(x_input)

# Print the output
print(output)
