import torch

class BMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initializing the model
model = BMMModel()

# Setting the configuration to True
torch._inductor.config.decompose_mem_bound_mm = True

# Generating input tensors
# Input tensor 1 with shape (10240, 32, 32)
input1 = torch.randn(10240, 32, 32, device='cuda')  # Ensure it's on the same device
# Input tensor 2 with shape (10240, 32, 32)
input2 = torch.randn(10240, 32, 32, device='cuda')  # Ensure it's on the same device

# Running the model
output = model(input1, input2)

# Verifying the output
print(output.shape)  # Should print: torch.Size([10240, 32, 32])
