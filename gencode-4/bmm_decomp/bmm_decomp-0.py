import torch

class BMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initialize the model
model = BMMModel()

# Set the configuration for decomposition
torch._inductor.config.decompose_mem_bound_mm = True

# Generate appropriate input tensors
# Both tensors need to be 3D, with input1 having a first dimension >= 10240
# and at least two of the other dimensions <= 32.

input1 = torch.randn(10240, 16, 32)  # Shape: (10240, 16, 32)
input2 = torch.randn(10240, 32, 16)  # Shape: (10240, 32, 16)

# Ensure both tensors are on the same device (CPU in this case)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Forward pass
output = model(input1, input2)

# Output shape
print(output.shape)  # Should be (10240, 16, 16)
