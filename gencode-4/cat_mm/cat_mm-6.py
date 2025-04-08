import torch

class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super(MatrixConcatModel, self).__init__()
        # Define the sizes for matrix multiplication
        self.input1_size = (4, 5)  # Example size for input1 (4x5 matrix)
        self.input2_size = (5, 3)  # Example size for input2 (5x3 matrix)

    def forward(self, input1, input2, additional_tensor):
        # Perform matrix multiplication
        t1 = torch.mm(input1, input2)
        
        # Concatenate the result of the matrix multiplication with additional_tensor along dimension 1
        t2 = torch.cat((t1, additional_tensor), dim=1)
        
        return t2

# Initialize the model
model = MatrixConcatModel()

# Inputs to the model
input1 = torch.randn(4, 5)  # Random tensor of shape (4, 5)
input2 = torch.randn(5, 3)  # Random tensor of shape (5, 3)
additional_tensor = torch.randn(4, 2)  # Additional tensor of shape (4, 2) for concatenation

# Getting the output from the model
output = model(input1, input2, additional_tensor)

# Print the output shape
print("Output shape:", output.shape)
