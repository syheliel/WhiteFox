import torch

# Assume torch._inductor.config.decompose_mem_bound_mm is True
# Setting up the model
class AddMMModel(torch.nn.Module):
    def __init__(self):
        super(AddMMModel, self).__init__()

    def forward(self, input_tensor, mat1, mat2):
        output = torch.addmm(input_tensor, mat1, mat2)
        return output

# Initializing the model
model = AddMMModel()

# Generating input tensors
input_tensor = torch.randn(10240, 32, device='cuda')
mat1 = torch.randn(10240, 32, device='cuda')
mat2 = torch.randn(32, 32, device='cuda')

# Forward pass
output = model(input_tensor, mat1, mat2)
