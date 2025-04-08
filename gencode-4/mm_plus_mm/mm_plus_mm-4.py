import torch

class MatrixModel(torch.nn.Module):
    def __init__(self, input_size1, input_size2, input_size3):
        super(MatrixModel, self).__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.input_size3 = input_size3
        
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # Matrix multiplication between input1 and input2
        t2 = torch.mm(input3, input4)  # Matrix multiplication between input3 and input4
        t3 = t1 + t2                   # Addition of the results of the two matrix multiplications
        return t3

# Initializing the model with specific input sizes
input_size1 = (5, 3)  # Example input size for input1 (5 rows, 3 columns)
input_size2 = (3, 4)  # Example input size for input2 (3 rows, 4 columns)
input_size3 = (5, 4)  # Example input size for input3 (5 rows, 4 columns)
input_size4 = (4, 2)  # Example input size for input4 (4 rows, 2 columns)

model = MatrixModel(input_size1, input_size2, input_size3)

# Generating input tensors
input1 = torch.randn(input_size1)  # Random tensor of shape (5, 3)
input2 = torch.randn(input_size2)  # Random tensor of shape (3, 4)
input3 = torch.randn(input_size3)  # Random tensor of shape (5, 4)
input4 = torch.randn(input_size4)  # Random tensor of shape (4, 2)

# Performing a forward pass
output = model(input1, input2, input3, input4)
print(output)
