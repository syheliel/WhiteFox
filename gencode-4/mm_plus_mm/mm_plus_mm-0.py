import torch

# Define the model
class MatrixModel(torch.nn.Module):
    def __init__(self, input_size1, input_size2, output_size):
        super(MatrixModel, self).__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.output_size = output_size

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # Matrix multiplication between input1 and input2
        t2 = torch.mm(input3, input4)  # Matrix multiplication between input3 and input4
        t3 = t1 + t2  # Addition of the results of the two matrix multiplications
        return t3

# Initialize the model
input_size1 = 4  # Number of columns in input1
input_size2 = 3  # Number of rows in input2
output_size = 5  # Number of columns in the output
m = MatrixModel(input_size1, input_size2, output_size)

# Generate input tensors for the model
input1 = torch.randn(2, input_size1)  # Batch of 2 with shape (2, 4)
input2 = torch.randn(input_size1, input_size2)  # Shape (4, 3)
input3 = torch.randn(2, input_size2)  # Batch of 2 with shape (2, 3)
input4 = torch.randn(input_size2, output_size)  # Shape (3, 5)

# Forward pass through the model
output = m(input1, input2, input3, input4)

# Print the output
print(output)
