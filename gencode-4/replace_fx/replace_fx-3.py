import torch

class CustomModel(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.linear = torch.nn.Linear(10, 10)  # A simple linear layer for demonstration

    def forward(self, x):
        t1 = torch.nn.functional.dropout(x, p=self.dropout_prob)  # Apply dropout to the input tensor
        t2 = torch.rand_like(x)  # Generate a tensor with the same size as input_tensor filled with random numbers
        return t1 + t2  # Return the sum of the two tensors

# Initializing the model
model = CustomModel()

# Generating an input tensor for the model
input_tensor = torch.randn(1, 10)  # Example input tensor with shape (1, 10)

# Forward pass through the model
output = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output)
