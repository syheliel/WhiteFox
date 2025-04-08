import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        # Apply a linear transformation to the input tensor
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function
        return t2

# Initializing the model
input_size = 20  # Example input size
output_size = 1  # Example output size for binary classification
model = SimpleModel(input_size, output_size)

# Generating inputs to the model
input_tensor = torch.randn(5, input_size)  # Batch size of 5 and input size of 20
output = model(input_tensor)

# Print the output
print(output)
