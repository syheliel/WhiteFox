import torch

# Model Definition
class MatrixModel(torch.nn.Module):
    def __init__(self, input_size, mat1_size, mat2_size):
        super().__init__()
        # Initialize matrices for multiplication
        self.mat1 = torch.nn.Parameter(torch.randn(mat1_size, input_size))
        self.mat2 = torch.nn.Parameter(torch.randn(mat2_size, mat1_size))
        self.input_size = input_size

    def forward(self, x):
        # Perform matrix multiplication and add to the input
        t1 = torch.addmm(x, self.mat1, self.mat2)
        # Concatenate the result along the specified dimension (0 in this case)
        t2 = torch.cat([t1], dim=0)
        return t2

# Model Parameters
input_size = 10
mat1_size = 5
mat2_size = 3

# Initializing the model
model = MatrixModel(input_size, mat1_size, mat2_size)

# Inputs to the model (batch size of 4 and input size of 10)
x_input = torch.randn(4, input_size)

# Forward pass
output = model(x_input)

# Print output shape
print(output.shape)
