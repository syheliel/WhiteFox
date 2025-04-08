import torch

class MatrixModel(torch.nn.Module):
    def __init__(self):
        super(MatrixModel, self).__init__()

    def forward(self, input, mat1, mat2):
        output = torch.addmm(input, mat1, mat2)
        return output

# Initializing the model
model = MatrixModel()

# Inputs to the model
input_tensor = torch.randn(10240, 32, device='cuda')  # Input tensor
mat1_tensor = torch.randn(10240, 32, device='cuda')   # mat1 tensor
mat2_tensor = torch.randn(32, 32, device='cuda')       # mat2 tensor

# Forward pass
output_tensor = model(input_tensor, mat1_tensor, mat2_tensor)

# Displaying the output shape
print(output_tensor.shape)
