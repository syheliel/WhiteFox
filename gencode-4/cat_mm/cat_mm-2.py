import torch

# Define the model
class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super(MatrixConcatModel, self).__init__()
        # Define the dimensions for the input tensors
        self.input1_size = (4, 3)  # 4 rows, 3 columns
        self.input2_size = (3, 2)   # 3 rows, 2 columns

    def forward(self, input1, input2, additional_tensor):
        # Matrix multiplication
        t1 = torch.mm(input1, input2)  # Result will have shape (4, 2)
        
        # Concatenation along dimension 1 (columns)
        t2 = torch.cat((t1, additional_tensor), dim=1)  # additional_tensor should have shape (4, N)

        return t2

# Initializing the model
model = MatrixConcatModel()

# Creating the input tensors
input1 = torch.randn(4, 3)  # Random tensor of shape (4, 3)
input2 = torch.randn(3, 2)   # Random tensor of shape (3, 2)
additional_tensor = torch.randn(4, 5)  # Random tensor of shape (4, 5) for concatenation

# Pass the inputs through the model
output = model(input1, input2, additional_tensor)

print("Input1:", input1)
print("Input2:", input2)
print("Additional Tensor:", additional_tensor)
print("Output:", output)
