import torch

# Assuming torch._inductor.config.decompose_mem_bound_mm is True

class AddMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, mat1, mat2):
        # Perform matrix multiplication and add to input_tensor
        output = torch.addmm(input_tensor, mat1, mat2)
        return output

# Initializing the model
model = AddMMModel()

# Generate input tensors
input_tensor = torch.randn(10240, 32, device='cuda')
mat1 = torch.randn(10240, 32, device='cuda')
mat2 = torch.randn(32, 32, device='cuda')

# Getting the output from the model
output = model(input_tensor, mat1, mat2)

# Print the shape of the output
print(output.shape)
