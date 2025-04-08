import torch

# Model Definition
class MatrixModel(torch.nn.Module):
    def __init__(self, input_size1, input_size2, input_size3):
        super(MatrixModel, self).__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.input_size3 = input_size3

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # Matrix multiplication between input1 and input2
        t2 = torch.mm(input3, input4)  # Matrix multiplication between input3 and input4
        t3 = t1 + t2  # Addition of the results of the two matrix multiplications
        return t3

# Initializing the model with specified dimensions
input_size1 = (4, 3)  # 4 rows, 3 columns for input1
input_size2 = (3, 5)  # 3 rows, 5 columns for input2
input_size3 = (4, 3)  # 4 rows, 3 columns for input3
input_size4 = (3, 5)  # 3 rows, 5 columns for input4
model = MatrixModel(input_size1, input_size2, input_size3)

# Creating input tensors
input1 = torch.randn(input_size1)  # Random tensor for input1
input2 = torch.randn(input_size2)  # Random tensor for input2
input3 = torch.randn(input_size3)  # Random tensor for input3
input4 = torch.randn(input_size4)  # Random tensor for input4

# Forward pass through the model
output = model(input1, input2, input3, input4)

# Print the output
print(output)
