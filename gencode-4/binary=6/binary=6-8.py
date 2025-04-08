import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.other = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Subtract this tensor from the output

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract 'other' from the output of the linear transformation
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x_input = torch.randn(1, 10)  # Batch size of 1 and 10 input features
output = m(x_input)
print("Input Tensor:\n", x_input)
print("Output Tensor:\n", output)
