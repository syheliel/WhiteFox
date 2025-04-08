import torch

class BMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initializing the model
bmm_model = BMMModel()

# Make sure the configuration is set to True
torch._inductor.config.decompose_mem_bound_mm = True

# Generate input tensors
# The first tensor has shape (10240, 32, 32)
input1 = torch.randn(10240, 32, 32)  # Shape meets the requirements
# The second tensor has shape (10240, 32, 32)
input2 = torch.randn(10240, 32, 32)  # Shape meets the requirements

# Ensure both tensors are on the same device (CPU)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Perform the forward pass
output = bmm_model(input1, input2)

# Output the shape of the result
print(output.shape)  # Should be (10240, 32, 32)
