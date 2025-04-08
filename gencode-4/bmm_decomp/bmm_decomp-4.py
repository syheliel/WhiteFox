import torch

# Ensure the required configuration is set
torch._inductor.config.decompose_mem_bound_mm = True

class BMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initialize the model
model = BMMModel()

# Generate input tensors
# Input tensor 1: shape (10240, 32, 32)
input1 = torch.randn(10240, 32, 32)

# Input tensor 2: shape (10240, 32, 32)
input2 = torch.randn(10240, 32, 32)

# Ensure both input tensors are on the same device (CPU in this case)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Forward pass through the model
output = model(input1, input2)

# Print the output shape
print(output.shape)  # Expected shape: (10240, 32, 32)
