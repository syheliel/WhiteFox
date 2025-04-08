import torch

# Model
class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super(MatrixConcatModel, self).__init__()
        # Defining layers for input tensors
        self.linear1 = torch.nn.Linear(10, 5)  # First input tensor will be of shape (batch_size, 10)
        self.linear2 = torch.nn.Linear(8, 5)   # Second input tensor will be of shape (batch_size, 8)

    def forward(self, input1, input2):
        # Matrix multiplication between the two input tensors
        t1 = torch.mm(input1, input2.t())  # Note: input2 is transposed to match dimensions
        
        # Concatenation of the result with additional tensors
        # For demonstration, we'll concatenate with the original inputs
        t2 = torch.cat((t1, input1, input2), dim=1)  # Concatenate along the feature dimension
        
        return t2

# Initializing the model
model = MatrixConcatModel()

# Inputs to the model
input1 = torch.randn(3, 10)  # Batch size of 3, feature size of 10
input2 = torch.randn(3, 8)    # Batch size of 3, feature size of 8
output = model(input1, input2)

# Print output shape
print(output.shape)  # Output shape will be (3, 5 + 10 + 8) = (3, 23)
