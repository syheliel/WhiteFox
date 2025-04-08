import torch

class BMMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initializing the model
model = BMMModel()

# Input tensors for the model
# Create input tensors according to the specified conditions
input1 = torch.randn(10240, 32, 32)  # 3D tensor with first dimension >= 10240, second and third <= 32
input2 = torch.randn(10240, 32, 32)  # 3D tensor with the same first dimension and compatible for bmm

# Ensure both input tensors are on the same device
input1 = input1.to('cuda')  # Assuming CUDA is available
input2 = input2.to('cuda')

# Make sure `decompose_mem_bound_mm` is set to True
torch._inductor.config.decompose_mem_bound_mm = True

# Running the model
output = model(input1, input2)
