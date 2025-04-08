import torch

class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)  # First linear layer
        self.linear2 = torch.nn.Linear(5, 5)   # Second linear layer

    def forward(self, input1, input2):
        # Matrix multiplication
        t1 = torch.mm(input1, input2)  # t1 = input1 @ input2
        # Concatenation
        t2 = torch.cat((t1, self.linear1(input1), self.linear2(input2)), dim=1)  # Concatenate along dimension 1
        return t2

# Initializing the model
model = MatrixConcatModel()

# Inputs to the model
input1 = torch.randn(4, 10)  # Batch size of 4, 10 features
input2 = torch.randn(10, 5)   # 10 features, 5 output features
output = model(input1, input2)

print(output)
