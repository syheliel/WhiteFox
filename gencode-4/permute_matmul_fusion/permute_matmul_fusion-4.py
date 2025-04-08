import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, input_tensor_A, input_tensor_B):
        # Permuting the last two dimensions of input_tensor_A
        t1 = input_tensor_A.permute(0, 2, 1)  # Assuming input_tensor_A is of shape (batch_size, channels, height)
        # Permuting the last two dimensions of input_tensor_B
        t2 = input_tensor_B.permute(0, 2, 1)  # Assuming input_tensor_B is of shape (batch_size, channels, height)
        # Perform batch matrix multiplication
        t3 = torch.bmm(t1, t2)  # (batch_size, height_A, height_B)
        return t3

# Initialize the model
model = MyModel()

# Generate input tensors
input_tensor_A = torch.randn(4, 3, 5)  # Example shape (batch_size=4, channels=3, height=5)
input_tensor_B = torch.randn(4, 3, 5)  # Example shape (batch_size=4, channels=3, height=5)

# Forward pass through the model
output = model(input_tensor_A, input_tensor_B)
