import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
        self.other = torch.tensor(1.5)  # Scalar to subtract from the linear output

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 - self.other  # Subtract 'other' from the output of the linear transformation
        return t2

# Initializing the model
model = LinearModel()

# Inputs to the model
x_input = torch.randn(1, 10)  # Input tensor with batch size 1 and 10 features
output = model(x_input)

print("Input Tensor:")
print(x_input)
print("Output Tensor:")
print(output)
