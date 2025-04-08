import torch

class BMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        # Perform batch matrix multiplication
        output = torch.bmm(input1, input2)
        return output

# Initializing the model
model = BMMModel()

# Setting the configuration to True for the decomposing condition
torch._inductor.config.decompose_mem_bound_mm = True

# Generating input tensors that meet the specified conditions
# Input shapes: (batch_size, M, K) and (batch_size, K, N)
# Let's say batch_size = 10240, M = 16, K = 32, N = 32
batch_size = 10240
M = 16
K = 32
N = 32

# Create input tensors on the same device (CPU in this case)
input1 = torch.randn(batch_size, M, K)  # (10240, 16, 32)
input2 = torch.randn(batch_size, K, N)  # (10240, 32, 32)

# Pass the input tensors through the model
output = model(input1, input2)

# Print the output shape
print(output.shape)  # Should be (10240, 16, 32)
