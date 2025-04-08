import torch

# Model
class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)  # A linear layer for transformation

    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)  # Matrix multiplication between two input tensors
        t2 = torch.cat((t1, input1), dim=1)  # Concatenate result of matrix multiplication with input1 along dimension 1
        return t2

# Initializing the model
model = MatrixModel()

# Inputs to the model
input1 = torch.randn(4, 5)  # Batch size of 4, 5 features
input2 = torch.randn(5, 3)  # 5 features, 3 outputs
output = model(input1, input2)

print(output)
