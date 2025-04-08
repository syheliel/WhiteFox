import torch

class MatrixConcatModel(torch.nn.Module):
    def __init__(self, input_dim, mat1_dim, mat2_dim):
        super().__init__()
        # Define the matrices for the matrix multiplication
        self.mat1 = torch.nn.Parameter(torch.randn(mat1_dim, input_dim))
        self.mat2 = torch.nn.Parameter(torch.randn(mat2_dim, mat1_dim))
        
    def forward(self, input_tensor):
        # Perform matrix multiplication and add to input
        t1 = torch.addmm(input_tensor, self.mat1, self.mat2)
        # Concatenate the result along dimension 1 (channels)
        t2 = torch.cat([t1], dim=1)
        return t2

# Initialize the model with specific dimensions
input_dim = 4  # Dimensionality of the input tensor
mat1_dim = 3    # Rows of mat1
mat2_dim = 2    # Rows of mat2
model = MatrixConcatModel(input_dim, mat1_dim, mat2_dim)

# Generate an input tensor
input_tensor = torch.randn(1, input_dim)

# Get the output from the model
output = model(input_tensor)
print(output)
