import torch

# Model Definition
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Linear layer with input size 10 and output size 5
        self.linear = torch.nn.Linear(10, 5)
        # Define the constant value to subtract
        self.other = 1.0

    def forward(self, x):
        t1 = self.linear(x)      # Apply linear transformation
        t2 = t1 - self.other     # Subtract 'other' from the linear output
        t3 = torch.relu(t2)      # Apply ReLU activation
        return t3

# Initializing the model
model = MyModel()

# Create an input tensor of shape (1, 10)
input_tensor = torch.randn(1, 10)

# Get the output from the model
output = model(input_tensor)

# Print the input and output tensors
print("Input Tensor:\n", input_tensor)
print("Output Tensor:\n", output)
