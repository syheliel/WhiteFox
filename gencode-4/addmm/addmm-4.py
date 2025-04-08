import torch

class MatrixModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MatrixModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, input1, input2, inp):
        t1 = torch.mm(input1, input2)  # Perform matrix multiplication
        t2 = t1 + inp                   # Add the input tensor 'inp'
        return t2

# Initializing the model with input size 4 and output size 3
model = MatrixModel(input_size=4, output_size=3)

# Creating input tensors
input1 = torch.randn(2, 4)  # Batch size of 2, input size of 4
input2 = torch.randn(4, 3)  # Input2 size matches input1's second dimension and has output size of 3
inp = torch.randn(2, 3)      # The tensor to add, must match the output size of the matrix multiplication

# Forward pass
output = model(input1, input2, inp)

print("Output Tensor:")
print(output)
