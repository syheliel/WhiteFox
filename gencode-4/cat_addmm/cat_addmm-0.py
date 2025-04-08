import torch
import torch.nn as nn

class MatrixAddConcatModel(nn.Module):
    def __init__(self, input_size, mat1_size, mat2_size):
        super(MatrixAddConcatModel, self).__init__()
        self.mat1 = nn.Parameter(torch.randn(mat1_size, input_size))  # Random weights for mat1
        self.mat2 = nn.Parameter(torch.randn(mat2_size, mat1_size))  # Random weights for mat2
        self.input_size = input_size

    def forward(self, x):
        # Perform matrix multiplication of mat1 and mat2 and add it to the input
        t1 = torch.addmm(x, self.mat1, self.mat2)
        
        # Concatenate the result along dimension 1
        t2 = torch.cat([t1], dim=1)
        
        return t2

# Initializing the model
input_size = 4  # Size of the input features
mat1_size = 3   # Size of the first matrix for multiplication
mat2_size = 2   # Size of the second matrix for multiplication
model = MatrixAddConcatModel(input_size, mat1_size, mat2_size)

# Inputs to the model
x_input = torch.randn(1, input_size)  # Batch size of 1, input feature size of 4
output = model(x_input)

print("Output shape:", output.shape)
