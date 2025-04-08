import torch

class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initializing the model
model = MatrixMultiplicationModel()

# Configuring to allow decomposition of memory bound matrix multiplications
torch._inductor.config.decompose_mem_bound_mm = True

# Creating input tensors
# Ensure the first dimension of input1 is >= 10240 and the dimensions of input2 are < 32
input1 = torch.randn(10240, 64)  # 2D tensor with shape (10240, 64)
input2 = torch.randn(64, 16)      # 2D tensor with shape (64, 16)

# Ensuring both tensors are on the same device (default is CPU)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Performing matrix multiplication
output = model(input1, input2)

# Verify the output shape
print(output.shape)  # Should be (10240, 16)
