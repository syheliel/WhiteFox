import torch

# Define a model class using PyTorch
class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()
        # Initialize the model's parameters
        self.mat1 = torch.randn(10240, 32, device='cuda', requires_grad=True)
        self.mat2 = torch.randn(32, 32, device='cuda', requires_grad=True)

    def forward(self, input_tensor):
        # Perform the matrix multiplication with addition
        output = torch.addmm(input_tensor, self.mat1, self.mat2)
        return output

# Initialize the model
model = MatrixMultiplicationModel()

# Create the input tensor
input_tensor = torch.randn(10240, 32, device='cuda')  # Ensure it is on the same device as mat1 and mat2

# Perform a forward pass through the model
output = model(input_tensor)

# Print output shape
print("Output shape:", output.shape)

input_tensor = torch.randn(10240, 32, device='cuda')
