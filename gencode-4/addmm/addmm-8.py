import torch

# Define the model
class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2, inp):
        t1 = torch.mm(input1, input2)  # Perform matrix multiplication
        t2 = t1 + inp  # Add the result of matrix multiplication to another tensor 'inp'
        return t2

# Initializing the model
model = MatrixModel()

# Generating input tensors
# Assuming the size of input1 is (batch_size, n) and input2 is (n, m)
# Here, we will define them with random values
batch_size = 1
n = 4  # Number of columns in input1 and rows in input2
m = 3  # Number of columns in input2

input1 = torch.randn(batch_size, n)  # Shape (1, 4)
input2 = torch.randn(n, m)            # Shape (4, 3)
inp = torch.randn(batch_size, m)      # Shape (1, 3)

# Forward pass
output = model(input1, input2, inp)

# Print the output
print(output)
