import torch

# Model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size, other):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.other = other

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation to the input tensor
        t2 = t1 - self.other  # Subtract 'other' from the output of the linear transformation
        t3 = torch.relu(t2)   # Apply the ReLU activation function to the result
        return t3

# Initialize the model
input_size = 10   # Size of the input features
output_size = 5   # Size of the output features
other_value = 1.0 # The value to subtract from the linear output
model = SimpleModel(input_size, output_size, other_value)

# Input tensor for the model
x = torch.randn(1, input_size)  # Batch size of 1 and input size of 10
output = model(x)

print("Input Tensor:", x)
print("Output Tensor:", output)
