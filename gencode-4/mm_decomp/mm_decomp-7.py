import torch

# Define the model
class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initialize the model
model = MatrixMultiplicationModel()

# Set the configuration to allow decomposition for memory bound matrix multiplication
torch._inductor.config.decompose_mem_bound_mm = True

# Create input tensors
# Ensure the first tensor has a first dimension >= 10240 and is 2-dimensional
# Ensure the second tensor has dimensions < 32
input1 = torch.randn(10240, 16)  # 10240 x 16 matrix
input2 = torch.randn(16, 32)      # 16 x 32 matrix

# Ensure both tensors are on the same device
input1 = input1.to('cpu')  # Assuming using CPU
input2 = input2.to('cpu')  # Both tensors on CPU

# Forward pass
output = model(input1, input2)

# Output the result
print(output.shape)  # Should be (10240, 32)
