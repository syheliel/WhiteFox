import torch

class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initialize the model
model = MatrixMultiplicationModel()

# Set the configuration for decomposing memory-bound matrix multiplication
torch._inductor.config.decompose_mem_bound_mm = True

# Generate input tensors
# Input tensor 1: Shape (10240, 64)
input1 = torch.randn(10240, 64)

# Input tensor 2: Shape (64, 32)
input2 = torch.randn(64, 32)

# Ensure both tensors are on the same device (CPU in this case)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Perform matrix multiplication
output = model(input1, input2)

# Print the output shape
print(output.shape)  # Should be (10240, 32)
