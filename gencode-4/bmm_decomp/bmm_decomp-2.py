import torch

class BMMModel(torch.nn.Module):
    def __init__(self):
        super(BMMModel, self).__init__()

    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initializing the model
bmm_model = BMMModel()

# Setting the configuration for the model to reach the `return True` line
torch._inductor.config.decompose_mem_bound_mm = True

# Generating input tensors that meet the specified conditions
# Input tensor shapes
# input1: (10240, 32, 32)
# input2: (10240, 32, 32)
input1 = torch.randn(10240, 32, 32, device='cuda')  # Change to 'cuda' for GPU
input2 = torch.randn(10240, 32, 32, device='cuda')  # Change to 'cuda' for GPU

# Performing the forward pass
output = bmm_model(input1, input2)

# Print output shape to verify
print(output.shape)
