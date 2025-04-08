import torch

# Ensure that the necessary configuration is set
# torch._inductor.config.decompose_mem_bound_mm = True  # Uncomment this in your environment

class AddMMModel(torch.nn.Module):
    def __init__(self):
        super(AddMMModel, self).__init__()
        # Initialize mat1 and mat2 as parameters
        self.mat1 = torch.nn.Parameter(torch.randn(10240, 32, device='cuda'))
        self.mat2 = torch.nn.Parameter(torch.randn(32, 32, device='cuda'))

    def forward(self, input_tensor):
        output = torch.addmm(input_tensor, self.mat1, self.mat2)
        return output

# Initialize the model
model = AddMMModel().to('cuda')

# Create input tensor
input_tensor = torch.randn(10240, 32, device='cuda')

# Get the output from the model
output = model(input_tensor)

print(output)

input_tensor = torch.randn(10240, 32, device='cuda')
