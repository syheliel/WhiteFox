import torch

class MatrixMultiplicationModel(torch.nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()

    def forward(self, input1, input2):
        output = torch.mm(input1, input2)
        return output

# Initialize the model
model = MatrixMultiplicationModel()

# Generate input tensors
# Ensure the first tensor has a first dimension >= 10240 and is 2D
input1 = torch.randn(10240, 64).to('cuda')  # Example tensor with shape (10240, 64)
# Ensure the second tensor has dimensions < 32 and is 2D
input2 = torch.randn(64, 32).to('cuda')  # Example tensor with shape (64, 32)

# Check if the tensors meet the conditions for decomposition
if (input1.device == input2.device and 
    input1.dim() == 2 and 
    input2.dim() == 2 and 
    input1.size(0) >= 10240 and 
    input2.size(0) < 32 and 
    input2.size(1) < 32):
    torch._inductor.config.decompose_mem_bound_mm = True
    output = model(input1, input2)
else:
    raise ValueError("Input tensors do not meet the required conditions.")
