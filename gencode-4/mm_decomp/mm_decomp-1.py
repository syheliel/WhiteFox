import torch

class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        # Perform matrix multiplication
        output = torch.mm(input1, input2)
        return output

# Initialize the model
model = MatrixMultiplicationModel()

# Creating input tensors that meet the specified conditions
# Input tensor 1: 10240x64 (first dimension >= 10240)
input1 = torch.randn(10240, 64, device='cuda')
# Input tensor 2: 64x30 (both dimensions < 32)
input2 = torch.randn(64, 30, device='cuda')

# Check the conditions before calling the model
# 1. Setting the configuration for decomposition
torch._inductor.config.decompose_mem_bound_mm = True

# 2. Both inputs are on the same device (CUDA)
# 3. Neither of the input tensors is symbolic (they are normal tensors)
# 4. Both input tensors are 2-dimensional
# 5. First dimension of input1 >= 10240
# 6. Both dimensions of input2 < 32

# Run the model
output = model(input1, input2)

# Output shape
print("Output shape:", output.shape)  # Should be (10240, 30)
