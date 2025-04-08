import torch

# Model definition
class MatrixModel(torch.nn.Module):
    def __init__(self):
        super(MatrixModel, self).__init__()

    def forward(self, input1, input2, inp):
        t1 = torch.mm(input1, input2)  # Perform matrix multiplication
        t2 = t1 + inp                   # Add the result to another tensor 'inp'
        return t2

# Initializing the model
model = MatrixModel()

# Inputs to the model
input1 = torch.randn(4, 3)  # Example tensor of shape (4, 3)
input2 = torch.randn(3, 5)  # Example tensor of shape (3, 5)
inp = torch.randn(4, 5)      # Example tensor of shape (4, 5) for addition

# Forward pass through the model
output = model(input1, input2, inp)

# Print output tensor
print(output)
