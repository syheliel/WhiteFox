import torch

class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the dimensions for the matrix
        self.input_size1 = 4  # Number of rows for the first matrix
        self.input_size2 = 3  # Number of columns for the first matrix and rows for the second matrix
        self.extra_tensor = torch.randn(4, 2)  # Additional tensor to concatenate

    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)  # Matrix multiplication
        t2 = torch.cat((t1, self.extra_tensor), dim=1)  # Concatenate along the second dimension
        return t2

# Initializing the model
model = MatrixModel()

# Inputs to the model
input_tensor1 = torch.randn(4, 3)  # Random tensor with shape (4, 3)
input_tensor2 = torch.randn(3, 5)  # Random tensor with shape (3, 5)

# Getting the output from the model
output = model(input_tensor1, input_tensor2)
print(output)
