import torch

class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initializing the model
model = MatrixMultiplicationModel()

# Set the configuration for decomposition
torch._inductor.config.decompose_mem_bound_mm = True

# Inputs to the model
# Create two input tensors that satisfy the conditions
input1 = torch.randn(10240, 64)  # First tensor with first dimension >= 10240
input2 = torch.randn(64, 32)      # Second tensor with both dimensions < 32

# Ensure both tensors are on the same device (CPU in this case)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Output from the model
output = model(input1, input2)

# Check the shape of the output
print(output.shape)  # This should print: torch.Size([10240, 32])
