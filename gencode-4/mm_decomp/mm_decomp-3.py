import torch

# Set the configuration to enable memory-bound matrix multiplication decomposition
torch._inductor.config.decompose_mem_bound_mm = True

# Define the model
class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()

    def forward(self, input1, input2):
        # Perform matrix multiplication
        output = torch.mm(input1, input2)
        return output

# Initialize the model
model = MatrixMultiplicationModel()

# Generate input tensors that meet the specified conditions
# Input 1: A tensor of shape (10240, 64) (first dimension >= 10240)
input1 = torch.randn(10240, 64)

# Input 2: A tensor of shape (64, 32) (both dimensions < 32)
input2 = torch.randn(64, 32)

# Ensure both tensors are on the same device (if using GPU, uncomment the following line)
# input1 = input1.to('cuda')
# input2 = input2.to('cuda')

# Compute the output
output = model(input1, input2)

# Display the output shape
print(output.shape)
