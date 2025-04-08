import torch

# Ensure that the decomposition configuration is set to True
torch._inductor.config.decompose_mem_bound_mm = True

class MatrixModel(torch.nn.Module):
    def __init__(self):
        super(MatrixModel, self).__init__()
        # Define a 2D tensor for mat1 and mat2
        self.mat1 = torch.randn(10240, 32, device='cuda', requires_grad=True)
        self.mat2 = torch.randn(32, 32, device='cuda', requires_grad=True)

    def forward(self, input_tensor):
        # Perform the matrix multiplication and addition
        output = torch.addmm(input_tensor, self.mat1, self.mat2)
        return output

# Initialize the model and move it to the CUDA device
model = MatrixModel().cuda()

# Generate the input tensor
input_tensor = torch.randn(10240, 32, device='cuda')

# Forward pass through the model
output = model(input_tensor)

# Print the shape of the output for verification
print(output.shape)
