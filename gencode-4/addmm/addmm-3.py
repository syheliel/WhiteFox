import torch

# Model
class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2, inp):
        t1 = torch.mm(input1, input2)  # Perform matrix multiplication
        t2 = t1 + inp                   # Add the result to the 'inp' tensor
        return t2

# Initializing the model
model = MatrixModel()

# Inputs to the model
input1 = torch.randn(4, 3)      # 4x3 matrix
input2 = torch.randn(3, 5)      # 3x5 matrix
inp = torch.randn(4, 5)         # 4x5 matrix to add to the result

# Forward pass
output = model(input1, input2, inp)

print("Output shape:", output.shape)
