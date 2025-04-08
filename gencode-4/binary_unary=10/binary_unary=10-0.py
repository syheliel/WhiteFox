import torch

class AnotherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5
        self.other = torch.randn(1, 5)  # Another tensor to be added (broadcastable)

    def forward(self, input_tensor):
        t1 = self.linear(input_tensor)  # Apply linear transformation
        t2 = t1 + self.other  # Add another tensor
        t3 = torch.relu(t2)  # Apply ReLU activation
        return t3

# Initializing the model
model = AnotherModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input features of size 10
output = model(input_tensor)

print("Output:", output)

input_tensor = torch.randn(1, 10)  # A random input tensor with shape (1, 10)
