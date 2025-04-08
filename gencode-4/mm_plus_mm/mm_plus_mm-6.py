import torch

class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define dimensions for matrix multiplication
        self.matrix1 = torch.nn.Parameter(torch.randn(4, 3))  # Shape (4, 3)
        self.matrix2 = torch.nn.Parameter(torch.randn(3, 5))  # Shape (3, 5)
        self.matrix3 = torch.nn.Parameter(torch.randn(4, 3))  # Shape (4, 3)
        self.matrix4 = torch.nn.Parameter(torch.randn(3, 5))  # Shape (3, 5)

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # Matrix multiplication of input1 and input2
        t2 = torch.mm(input3, input4)  # Matrix multiplication of input3 and input4
        t3 = t1 + t2                   # Addition of the results of the two matrix multiplications
        return t3

# Initializing the model
model = MatrixModel()

# Inputs to the model
input1 = torch.randn(4, 3)  # Example input tensor for first matrix multiplication
input2 = model.matrix2       # Using the parameter defined in the model
input3 = torch.randn(4, 3)  # Example input tensor for second matrix multiplication
input4 = model.matrix4       # Using the parameter defined in the model

# Forward pass
output = model(input1, input2, input3, input4)

# Display the output
print(output)
