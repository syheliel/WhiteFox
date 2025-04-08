import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer with input size 10 and output size 5
        self.other = 3.0  # The value to subtract from the linear transformation output

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract 'other' from the output of the linear transformation
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initialize the model
model = SimpleModel()

# Generate an input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input size of 10

# Forward pass
output = model(input_tensor)

# Displaying the output
print("Output of the model:", output)
