import torch

class BMMModel(torch.nn.Module):
    def __init__(self):
        super(BMMModel, self).__init__()

    def forward(self, input1, input2):
        output = torch.bmm(input1, input2)
        return output

# Initialize the model
bmm_model = BMMModel()

# Input tensor creation
# Input dimensions must meet the specified conditions:
# - First dimension >= 10240
# - At least two of the following dimensions <= 32:
#     - Second dimension of input1
#     - Third dimension of input1
#     - Third dimension of input2

input1 = torch.randn(10240, 32, 32)  # Shape: (10240, 32, 32)
input2 = torch.randn(10240, 32, 32)  # Shape: (10240, 32, 32)

# Ensure that both inputs are on the same device (CPU in this case)
input1 = input1.to('cpu')
input2 = input2.to('cpu')

# Run the model
output = bmm_model(input1, input2)

# Output shape will be (10240, 32, 32)
print(output.shape)  # Should print: torch.Size([10240, 32, 32])
