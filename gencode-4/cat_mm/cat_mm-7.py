import torch

# Model definition
class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super(MatrixConcatModel, self).__init__()
        self.linear1 = torch.nn.Linear(4, 3)  # Linear layer for first input
        self.linear2 = torch.nn.Linear(4, 3)  # Linear layer for second input

    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)  # Matrix multiplication
        t2 = torch.cat((t1, input1), dim=1)  # Concatenate with input1 along dimension 1
        return t2

# Initializing the model
model = MatrixConcatModel()

# Inputs to the model
input1 = torch.randn(2, 4)  # Example input tensor of shape (2, 4)
input2 = torch.randn(4, 4)  # Example input tensor of shape (4, 4)

# Forward pass
output = model(input1, input2)

print("Input 1:", input1)
print("Input 2:", input2)
print("Output:", output)
