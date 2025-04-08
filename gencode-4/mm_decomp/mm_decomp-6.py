import torch

# Setting the configuration for decomposing memory-bound matrix multiplication
torch._inductor.config.decompose_mem_bound_mm = True

# Define the model
class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initializing the model
model = MatrixMultiplicationModel()

# Create input tensors that meet the specified conditions
# Ensure the first dimension of input1 is >= 10240 and the dimensions of input2 are < 32
input1 = torch.randn(10240, 128).to('cuda')  # First input tensor
input2 = torch.randn(128, 16).to('cuda')     # Second input tensor

# Perform the matrix multiplication
output = model(input1, input2)

print("Output shape:", output.shape)
