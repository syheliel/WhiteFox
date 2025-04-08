import torch

# Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions
        self.other_tensor = torch.randn(1, 5)  # A tensor to be added (must have the same shape as output of linear layer)

    def forward(self, x1):
        t1 = self.linear(x1)  # Apply a linear transformation
        t2 = t1 + self.other_tensor  # Add the other tensor
        t3 = torch.nn.functional.relu(t2)  # Apply the ReLU activation function
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)  # Input tensor with shape (1, 10)
output = m(x1)

print("Output:", output)
