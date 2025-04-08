import torch

# Model definition
class LinearReLUModel(torch.nn.Module):
    def __init__(self):
        super(LinearReLUModel, self).__init__()
        # Define a linear layer that takes an input of size 10 and outputs size 5
        self.linear = torch.nn.Linear(10, 5)
        self.other = 1.0  # The value to subtract from the linear transformation output

    def forward(self, x):
        t1 = self.linear(x)          # Apply a linear transformation
        t2 = t1 - self.other         # Subtract 'other' from the output of the linear transformation
        t3 = torch.relu(t2)          # Apply the ReLU activation function
        return t3

# Initializing the model
model = LinearReLUModel()

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1, with 10 features
output = model(input_tensor)

# Print the output
print(output)
