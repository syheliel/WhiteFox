import torch

# Model definition
class MatrixModel(torch.nn.Module):
    def __init__(self, input_size1, input_size2, output_size):
        super().__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.output_size = output_size

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # Matrix multiplication between input1 and input2
        t2 = torch.mm(input3, input4)  # Matrix multiplication between input3 and input4
        t3 = t1 + t2  # Addition of the results of the two matrix multiplications
        return t3

# Initialize the model
input_size1 = (4, 5)  # Example input dimensions for the first matrix multiplication
input_size2 = (5, 3)  # Example input dimensions for the second matrix multiplication
output_size = (4, 3)  # Expected output size

model = MatrixModel(input_size1[0], input_size2[0], output_size)

# Generating input tensors
input1 = torch.randn(input_size1)  # Random tensor of shape (4, 5)
input2 = torch.randn(input_size2)  # Random tensor of shape (5, 3)
input3 = torch.randn(input_size1)  # Random tensor of shape (4, 5) for the second multiplication
input4 = torch.randn(input_size2)  # Random tensor of shape (5, 3) for the second multiplication

# Forward pass
output = model(input1, input2, input3, input4)

# Displaying the output
print(output)
