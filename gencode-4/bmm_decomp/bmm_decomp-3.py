import torch

# Setting the configuration for the model to ensure it meets the `should_decompose_bmm` conditions
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
# First input tensor: shape (10240, 32, 32)
input1 = torch.randn(10240, 32, 32)

# Second input tensor: shape (10240, 32, 32)
input2 = torch.randn(10240, 32, 32)

# Ensure both inputs are on the same device
input1 = input1.to('cuda')  # Assuming CUDA is available
input2 = input2.to('cuda')

# Forward pass through the model
output = model(input1, input2)

# Printing the output shape
print("Output shape:", output.shape)
