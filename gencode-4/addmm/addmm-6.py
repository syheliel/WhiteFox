import torch

# Model definition
class MatrixModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, input1, input2, inp):
        t1 = torch.mm(input1, input2)  # Perform matrix multiplication on two input tensors
        t2 = t1 + inp                   # Add the result of the matrix multiplication to another tensor 'inp'
        return t2

# Initializing the model
input_dim = 4   # Example input dimension
output_dim = 3  # Example output dimension
model = MatrixModel(input_dim, output_dim)

# Input tensors
input1 = torch.randn(2, input_dim)  # Batch size of 2, input_dim columns
input2 = torch.randn(input_dim, output_dim)  # input_dim rows, output_dim columns
inp = torch.randn(2, output_dim)  # Same batch size as input1, output_dim columns

# Forward pass
output = model(input1, input2, inp)

# Displaying the output
print(output)
