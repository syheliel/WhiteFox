import torch

class MatrixModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MatrixModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input1, input2, inp):
        t1 = torch.mm(input1, input2)  # Perform matrix multiplication
        t2 = t1 + inp                   # Add the result to the tensor 'inp'
        return t2

# Initializing the model
input_size = 4  # For example, 4 features in input1
output_size = 3 # For example, resulting output will have 3 features
model = MatrixModel(input_size, output_size)

# Example inputs to the model
input1 = torch.randn(5, input_size)  # Batch of 5 samples, each with 4 features
input2 = torch.randn(input_size, output_size)  # Weight matrix for matrix multiplication
inp = torch.randn(5, output_size)    # Tensor to be added to the result

# Forward pass
output = model(input1, input2, inp)

print(output)
