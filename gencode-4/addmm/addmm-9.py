import torch

# Model
class MatrixModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input1, input2, inp):
        t1 = torch.mm(input1, input2)  # Perform matrix multiplication
        t2 = t1 + inp                   # Add the result to the 'inp' tensor
        return t2

# Initialize the model with specific dimensions
input_dim = 5
output_dim = 3
model = MatrixModel(input_dim, output_dim)

# Inputs to the model
input1 = torch.randn(10, input_dim)  # A batch of 10 samples
input2 = torch.randn(input_dim, output_dim)  # Matrix for multiplication
inp = torch.randn(10, output_dim)  # Tensor to be added

# Forward pass
output = model(input1, input2, inp)

print(output.shape)  # Should be (10, output_dim)
