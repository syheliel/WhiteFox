import torch

# Define the model
class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super(MatrixConcatModel, self).__init__()
        self.linear1 = torch.nn.Linear(10, 5)  # First linear transformation
        self.linear2 = torch.nn.Linear(5, 5)   # Second linear transformation

    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)  # Matrix multiplication
        t2 = self.linear1(t1)          # Apply first linear layer
        t3 = self.linear2(t2)          # Apply second linear layer
        t4 = torch.cat((t1, t3), dim=1)  # Concatenate along the feature dimension
        return t4

# Initializing the model
model = MatrixConcatModel()

# Create input tensors
input1 = torch.randn(3, 10)  # Batch of 3 samples, each with 10 features
input2 = torch.randn(10, 5)   # A matrix with dimensions compatible for multiplication with input1

# Forward pass through the model
output = model(input1, input2)

# Display the output shape
print("Output shape:", output.shape)
