import torch

class MatrixConcatModel(torch.nn.Module):
    def __init__(self, input_dim, mat1_dim, mat2_dim, concat_dim):
        super(MatrixConcatModel, self).__init__()
        # Define the matrices for matrix multiplication
        self.mat1 = torch.nn.Parameter(torch.randn(mat1_dim, mat2_dim))
        self.mat2 = torch.nn.Parameter(torch.randn(mat2_dim, input_dim))
        self.concat_dim = concat_dim

    def forward(self, input):
        # Perform matrix multiplication and add to input
        t1 = torch.addmm(input, input, self.mat1 @ self.mat2)
        # Concatenate the result along a specified dimension
        t2 = torch.cat([t1], dim=self.concat_dim)
        return t2

# Initializing the model with dimensions
input_dim = 4
mat1_dim = 4
mat2_dim = 4
concat_dim = 0  # Concatenation along the first dimension

model = MatrixConcatModel(input_dim, mat1_dim, mat2_dim, concat_dim)

# Generate a random input tensor
input_tensor = torch.randn(1, input_dim)

# Pass the input through the model
output = model(input_tensor)

# Output the result
print(output)
