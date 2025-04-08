import torch

# Ensure that the configuration for decompose_mem_bound_mm is set to True
torch._inductor.config.decompose_mem_bound_mm = True

class AddMMModel(torch.nn.Module):
    def __init__(self):
        super(AddMMModel, self).__init__()

    def forward(self, input_tensor, mat1, mat2):
        output = torch.addmm(input_tensor, mat1, mat2)
        return output

# Initialize the model
model = AddMMModel().to('cuda')  # Make sure the model is on the same device as the inputs

# Create the input tensors according to the specified conditions
input_tensor = torch.randn(10240, 32, device='cuda')  # Input tensor (2D)
mat1 = torch.randn(10240, 32, device='cuda')           # mat1 (2D, first dimension >= 10240)
mat2 = torch.randn(32, 32, device='cuda')               # mat2 (2D, both dimensions < 32)

# Forward pass
output = model(input_tensor, mat1, mat2)

print(output.shape)  # Output shape should be (10240, 32)
