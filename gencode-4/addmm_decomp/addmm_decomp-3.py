import torch

class AddMMModel(torch.nn.Module):
    def __init__(self):
        super(AddMMModel, self).__init__()

    def forward(self, input, mat1, mat2):
        # Using the addmm function to perform matrix multiplication and addition
        output = torch.addmm(input, mat1, mat2)
        return output

# Initializing the model
model = AddMMModel()

# Assume torch._inductor.config.decompose_mem_bound_mm is True
# Creating inputs for the model
input_tensor = torch.randn(10240, 32, device='cuda')  # Input tensor
mat1 = torch.randn(10240, 32, device='cuda')           # First matrix for multiplication
mat2 = torch.randn(32, 32, device='cuda')               # Second matrix for multiplication

# Getting the output from the model
output = model(input_tensor, mat1, mat2)

print(output.shape)  # To verify the output shape
