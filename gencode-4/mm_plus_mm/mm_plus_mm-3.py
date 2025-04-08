import torch

# Model definition
class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the dimensions for the input matrices
        self.input_size1 = (4, 3)  # For input1
        self.input_size2 = (3, 5)  # For input2
        self.input_size3 = (4, 2)  # For input3
        self.input_size4 = (2, 5)  # For input4

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # Matrix multiplication between input1 and input2
        t2 = torch.mm(input3, input4)  # Matrix multiplication between input3 and input4
        t3 = t1 + t2  # Addition of the results of the two matrix multiplications
        return t3

# Initializing the model
model = MatrixModel()

# Generating input tensors for the model
input1 = torch.randn(4, 3)  # Random tensor for input1
input2 = torch.randn(3, 5)  # Random tensor for input2
input3 = torch.randn(4, 2)  # Random tensor for input3
input4 = torch.randn(2, 5)  # Random tensor for input4

# Getting the output of the model
output = model(input1, input2, input3, input4)

# Displaying the output
print(output)
