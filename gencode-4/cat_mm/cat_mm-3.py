import torch

class MatrixConcatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)  # First linear layer
        self.linear2 = torch.nn.Linear(10, 5)  # Second linear layer

    def forward(self, input1, input2):
        # Matrix multiplication between two input tensors
        t1 = torch.mm(input1, input2.t())  # Transposing input2 for proper matrix multiplication
        
        # Concatenating the result with additional tensors (here we concatenate t1 with itself for demonstration)
        t2 = torch.cat((t1, t1), dim=1)  # Concatenate along the second dimension (features)
        
        return t2

# Initializing the model
model = MatrixConcatModel()

# Inputs to the model
input_tensor1 = torch.randn(3, 10)  # Batch size of 3, feature size of 10
input_tensor2 = torch.randn(3, 10)  # Another tensor with the same shape for multiplication

# Forward pass through the model
output = model(input_tensor1, input_tensor2)

print("Output shape:", output.shape)
