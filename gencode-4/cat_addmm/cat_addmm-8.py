import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self, input_dim, mat1_dim, mat2_dim):
        super(Model, self).__init__()
        self.mat1 = torch.nn.Parameter(torch.randn(mat1_dim, input_dim))  # mat1 for matrix multiplication
        self.mat2 = torch.nn.Parameter(torch.randn(mat2_dim, mat1_dim))  # mat2 for matrix multiplication

    def forward(self, input_tensor):
        t1 = torch.addmm(input_tensor, self.mat1, self.mat2)  # Matrix multiplication and addition
        t2 = torch.cat([t1], dim=1)  # Concatenate along dimension 1
        return t2

# Initialize the model with specific dimensions
input_dim = 4
mat1_dim = 3
mat2_dim = 2
model = Model(input_dim, mat1_dim, mat2_dim)

# Inputs to the model
input_tensor = torch.randn(1, input_dim)  # Batch size of 1 and input dimension of 4
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
