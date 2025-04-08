import torch

class MatrixModel(torch.nn.Module):
    def __init__(self, input_size1, input_size2, input_size3, input_size4):
        super(MatrixModel, self).__init__()
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.input_size3 = input_size3
        self.input_size4 = input_size4

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)  # First matrix multiplication
        t2 = torch.mm(input3, input4)  # Second matrix multiplication
        t3 = t1 + t2                    # Addition of results
        return t3

# Initialize the model with input sizes
input_size1 = 4  # Number of rows for input1
input_size2 = 5  # Number of columns for input1 and rows for input2
input_size3 = 4  # Number of rows for input3 (same as input1)
input_size4 = 5  # Number of columns for input3 and rows for input4 (same as input2)

model = MatrixModel(input_size1, input_size2, input_size3, input_size4)

# Create input tensors
input1 = torch.randn(input_size1, input_size2)  # Shape: (4, 5)
input2 = torch.randn(input_size2, input_size2)  # Shape: (5, 5)
input3 = torch.randn(input_size3, input_size4)  # Shape: (4, 5)
input4 = torch.randn(input_size4, input_size4)  # Shape: (5, 5)

# Forward pass through the model
output = model(input1, input2, input3, input4)

print("Output Shape:", output.shape)  # Output shape should be (4, 5)
print("Output:", output)
