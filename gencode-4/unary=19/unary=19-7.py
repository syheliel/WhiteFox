import torch

# Define the model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        t1 = self.linear(x)          # Apply a linear transformation
        t2 = torch.sigmoid(t1)      # Apply the sigmoid function
        return t2

# Initialize the model
input_size = 10  # Size of the input feature vector
output_size = 1  # Size of the output (for binary classification)
model = SimpleModel(input_size, output_size)

# Generate a random input tensor with a batch size of 1
input_tensor = torch.randn(1, input_size)

# Get the output from the model
output = model(input_tensor)

# Display the input and output
print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output)
