import torch

# Model definition
class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super(MatrixConcatModel, self).__init__()
        # Define an extra linear layer to generate a tensor for concatenation
        self.linear = torch.nn.Linear(10, 5)  # Transforming the input to an appropriate size for concatenation

    def forward(self, input1, input2):
        # Matrix multiplication
        t1 = torch.mm(input1, input2)
        
        # Generate an additional tensor to concatenate
        additional_tensor = self.linear(torch.randn(input1.size(0), 10))  # Random tensor for concatenation
        
        # Concatenate the result of matrix multiplication with the additional tensor along the last dimension
        t2 = torch.cat((t1, additional_tensor), dim=1)  # Concatenate along columns (dim=1)

        return t2

# Initializing the model
model = MatrixConcatModel()

# Inputs to the model
input1 = torch.randn(4, 10)  # Batch size of 4, 10 features
input2 = torch.randn(10, 6)   # Matrix compatible for multiplication (10x6)

# Forward pass
output = model(input1, input2)
print(output)
