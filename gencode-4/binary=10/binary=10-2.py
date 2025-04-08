import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer with input size 10 and output size 5

    def forward(self, x1, other):
        t1 = self.linear(x1)  # Apply linear transformation
        t2 = t1 + other       # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)  # Input tensor with shape (1, 10)
other = torch.randn(1, 5)  # Another tensor to add with shape (1, 5)
output = m(x1, other)

print(output)
