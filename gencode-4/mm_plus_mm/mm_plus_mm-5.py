import torch

# Model definition
class MatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize weights for the two matrix multiplications
        self.weight1 = torch.nn.Parameter(torch.randn(4, 3))  # Matrix for first multiplication (4x3)
        self.weight2 = torch.nn.Parameter(torch.randn(3, 2))  # Matrix for second multiplication (3x2)
        self.weight3 = torch.nn.Parameter(torch.randn(4, 3))  # Matrix for third multiplication (4x3)
        self.weight4 = torch.nn.Parameter(torch.randn(3, 2))  # Matrix for fourth multiplication (3x2)

    def forward(self, input1, input2, input3, input4):
        # Perform matrix multiplications
        t1 = torch.mm(input1, self.weight1)  # First matrix multiplication
        t2 = torch.mm(input3, self.weight3)  # Second matrix multiplication
        # Add the results of the two matrix multiplications
        t3 = t1 + t2
        return t3

# Initializing the model
model = MatrixModel()

# Generate input tensors for the model
input1 = torch.randn(5, 3)  # Batch of 5 samples, with 3 features each
input2 = torch.randn(5, 4)  # This input is not used in computation but defined as per the requirement
input3 = torch.randn(5, 3)  # Another batch of 5 samples, with 3 features each
input4 = torch.randn(5, 4)  # This input is not used in computation but defined as per the requirement

# Forward pass through the model
output = model(input1, input2, input3, input4)

# Print the output shape
print(output.shape)
