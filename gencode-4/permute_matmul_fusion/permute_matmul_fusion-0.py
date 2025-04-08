import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the last two dimensions of input_tensor_A
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A has shape (batch_size, dim1, dim2)
        # Permute the last two dimensions of input_tensor_B
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B has shape (batch_size, dim3, dim4)
        
        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # Resulting shape will be (batch_size, dim2, dim4)
        return t3

# Initializing the model
model = CustomModel()

# Inputs to the model
# Assume we have a batch size of 2, and dimensions are (2, 4, 3) for input_tensor_A and (2, 3, 5) for input_tensor_B
input_tensor_A = torch.randn(2, 4, 3)  # Shape: (batch_size, dim1, dim2)
input_tensor_B = torch.randn(2, 3, 5)  # Shape: (batch_size, dim3, dim4)

# Forward pass
output = model(input_tensor_A, input_tensor_B)

# Output shape
print(output.shape)  # Should print: (2, 4, 5)
