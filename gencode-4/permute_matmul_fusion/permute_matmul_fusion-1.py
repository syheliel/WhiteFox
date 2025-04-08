import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer for demonstration purposes
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, input_tensor_A, input_tensor_B):
        # Permute the first input tensor to swap the last two dimensions
        t1 = input_tensor_A.permute(0, 2, 1)  # Change shape to (batch_size, depth, height)
        # Permute the second input tensor to swap the last two dimensions
        t2 = input_tensor_B.permute(0, 2, 1)  # Change shape to (batch_size, depth, height)
        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)
        return t3

# Initializing the model
model = CustomModel()

# Generating input tensors
# Assume input_tensor_A and input_tensor_B are of shape (batch_size, height, depth)
batch_size = 2
height = 4
depth = 4
input_tensor_A = torch.randn(batch_size, height, depth)  # Shape: (2, 4, 4)
input_tensor_B = torch.randn(batch_size, height, depth)  # Shape: (2, 4, 4)

# Forward pass through the model
output = model(input_tensor_A, input_tensor_B)

print("Output shape:", output.shape)
