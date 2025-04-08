import torch

# Assume torch._inductor.config.decompose_mem_bound_mm is True

class AddMMModel(torch.nn.Module):
    def __init__(self):
        super(AddMMModel, self).__init__()
        
    def forward(self, input, mat1, mat2):
        # Perform matrix multiplication and add to input
        output = torch.addmm(input, mat1, mat2)
        return output

# Initializing the model
model = AddMMModel()

# Creating input tensors
input_tensor = torch.randn(10240, 32, device='cuda')  # Input tensor
mat1_tensor = torch.randn(10240, 32, device='cuda')   # mat1 tensor
mat2_tensor = torch.randn(32, 32, device='cuda')       # mat2 tensor

# Getting the model output
output_tensor = model(input_tensor, mat1_tensor, mat2_tensor)

# Print the output shape
print(output_tensor.shape)
