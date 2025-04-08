import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the input tensor A to swap the last two dimensions
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A is of shape (batch_size, height, width)
        # Permute the input tensor B to swap the last two dimensions
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B is of shape (batch_size, height, width)

        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)
        return t3

# Initializing the model
model = MyModel()

# Inputs to the model
# Let's assume we have a batch size of 2, and the input tensors have a shape of (batch_size, height, width)
input_tensor_A = torch.randn(2, 4, 3)  # Shape: (2, 4, 3)
input_tensor_B = torch.randn(2, 4, 3)  # Shape: (2, 4, 3)

# Getting the output from the model
output = model(input_tensor_A, input_tensor_B)

# Print the output shape
print(output.shape)  # Expected shape: (2, 3, 3)
