import torch

# Ensure the decompose_mem_bound_mm configuration is set to True
torch._inductor.config.decompose_mem_bound_mm = True

class BMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initializing the model
model = BMMModel()

# Generate input tensors that meet the specified conditions
# Input tensor 1: shape (10240, 32, 32)
# Input tensor 2: shape (10240, 32, 32)
input1 = torch.randn(10240, 32, 32)  # First dimension >= 10240, second and third dimensions <= 32
input2 = torch.randn(10240, 32, 32)  # Second tensor must match the first tensor's first dimension

# Ensure both tensors are on the same device (CPU in this case)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Forward pass through the model
output = model(input1, input2)

# Check the output shape
print("Output shape:", output.shape)
