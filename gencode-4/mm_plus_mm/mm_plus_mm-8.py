import torch

class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # Matrix multiplication between input1 and input2
        t2 = torch.mm(input3, input4)  # Matrix multiplication between input3 and input4
        t3 = t1 + t2  # Addition of the results of the two matrix multiplications
        return t3

# Initializing the model
model = MatrixModel()

# Inputs to the model
# Define input tensors with appropriate sizes for matrix multiplication
input1 = torch.randn(4, 3)  # First input matrix (4x3)
input2 = torch.randn(3, 5)  # Second input matrix (3x5)
input3 = torch.randn(4, 3)  # Third input matrix (4x3)
input4 = torch.randn(3, 5)  # Fourth input matrix (3x5)

# Forward pass through the model
output = model(input1, input2, input3, input4)

print("Output shape:", output.shape)  # Should print: Output shape: torch.Size([4, 5])
