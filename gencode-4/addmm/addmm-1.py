import torch

# Model
class MatrixModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(input_size, output_size))

    def forward(self, input1, inp):
        t1 = torch.mm(input1, self.weight)  # Matrix multiplication
        t2 = t1 + inp  # Add the input tensor
        return t2

# Initializing the model
input_size = 4  # Size of the first input tensor (number of columns)
output_size = 3  # Size of the output after matrix multiplication (number of rows)
model = MatrixModel(input_size, output_size)

# Inputs to the model
batch_size = 2  # Number of samples
input1 = torch.randn(batch_size, input_size)  # Random input tensor for matrix multiplication
inp = torch.randn(batch_size, output_size)  # Random tensor to add

# Forward pass
output = model(input1, inp)

# Printing the output
print("Output of the model:")
print(output)
