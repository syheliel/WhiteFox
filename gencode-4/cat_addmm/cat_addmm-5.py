import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self, input_size, mat1_size, mat2_size, concat_dim):
        super().__init__()
        self.input_size = input_size
        self.mat1 = torch.nn.Parameter(torch.randn(mat1_size, input_size))
        self.mat2 = torch.nn.Parameter(torch.randn(mat2_size, mat1_size))
        self.concat_dim = concat_dim

    def forward(self, x):
        t1 = torch.addmm(x, self.mat1, self.mat2)  # Perform matrix multiplication and add to input
        t2 = torch.cat([t1], dim=self.concat_dim)  # Concatenate along the specified dimension
        return t2

# Initializing the model with arbitrary sizes
input_size = 5
mat1_size = 3
mat2_size = 4
concat_dim = 1  # Concatenating along dimension 1

m = Model(input_size, mat1_size, mat2_size, concat_dim)

# Inputs to the model
x1 = torch.randn(2, input_size)  # Batch size of 2

# Get the output
__output__ = m(x1)

# Display the output
print(__output__)
