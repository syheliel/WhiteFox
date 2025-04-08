import torch

# Model definition
class MatrixModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MatrixModel, self).__init__()
        # Define input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input1, input2, inp):
        # Perform matrix multiplication on input1 and input2
        t1 = torch.mm(input1, input2)
        # Add the result of the matrix multiplication to another tensor 'inp'
        t2 = t1 + inp
        return t2

# Initializing the model with specified input and output dimensions
input_dim = 4  # Example input dimension
output_dim = 3  # Example output dimension
model = MatrixModel(input_dim, output_dim)

# Inputs to the model
input1 = torch.randn(5, input_dim)  # Batch size of 5
input2 = torch.randn(input_dim, output_dim)  # Matrix to multiply with
inp = torch.randn(5, output_dim)  # Tensor to add to the result

# Forward pass through the model
output = model(input1, input2, inp)

# Display the output
print(output)
