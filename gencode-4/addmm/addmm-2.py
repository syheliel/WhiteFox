import torch

class MatrixModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Define a learnable weight matrix for the matrix multiplication
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim))

    def forward(self, input1, inp):
        t1 = torch.mm(input1, self.weight)  # Matrix multiplication
        t2 = t1 + inp  # Add the 'inp' tensor
        return t2

# Example initialization of the model
input_dim = 4   # Number of input features
output_dim = 3  # Number of output features
model = MatrixModel(input_dim, output_dim)

# Inputs to the model
input1 = torch.randn(2, input_dim)  # Batch of 2 samples, each with 'input_dim' features
inp = torch.randn(2, output_dim)     # 'inp' tensor to be added, must match output dimensions

# Forward pass
output = model(input1, inp)

print(output)
