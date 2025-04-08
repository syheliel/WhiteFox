import torch

class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initializing the model
model = MatrixMultiplicationModel()

# Setting the configuration for decomposition
torch._inductor.config.decompose_mem_bound_mm = True

# Generating input tensors
# Ensure the first dimension of input1 is >= 10240 and input2 is 2-dimensional with dimensions < 32
input1 = torch.randn(10240, 64).to('cuda')  # First tensor with shape (10240, 64)
input2 = torch.randn(64, 32).to('cuda')     # Second tensor with shape (64, 32)

# Performing matrix multiplication
output = model(input1, input2)
print(output.shape)  # Should print torch.Size([10240, 32])
