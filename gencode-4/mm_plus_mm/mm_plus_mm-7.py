import torch

# Define the model class
class MatrixModel(torch.nn.Module):
    def __init__(self, input_size1, input_size2, input_size3):
        super(MatrixModel, self).__init__()
        # Define the linear layers for matrix multiplication
        self.linear1 = torch.nn.Linear(input_size1, input_size2)
        self.linear2 = torch.nn.Linear(input_size3, input_size2)

    def forward(self, input1, input2, input3, input4):
        # Matrix multiplication between input1 and input2
        t1 = torch.mm(input1, input2)
        # Matrix multiplication between input3 and input4
        t2 = torch.mm(input3, input4)
        # Addition of the results of the two matrix multiplications
        t3 = t1 + t2
        return t3

# Initialize the model with specific input sizes
input_size1 = 4  # Number of features in input1
input_size2 = 3  # Number of features in input2 and output of linear layers
input_size3 = 4  # Number of features in input3 (should match input_size1 for mm)

model = MatrixModel(input_size1, input_size2, input_size3)

# Create example input tensors
input1 = torch.randn(2, input_size1)  # Batch size of 2
input2 = torch.randn(input_size1, input_size2)  # Shape must match for matrix multiplication
input3 = torch.randn(2, input_size3)  # Batch size of 2
input4 = torch.randn(input_size3, input_size2)  # Shape must match for matrix multiplication

# Forward pass through the model
output = model(input1, input2, input3, input4)

# Print the output
print(output)
