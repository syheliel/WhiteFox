import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from input size 10 to output size 5
        self.other_tensor = torch.randn(1, 5)  # A tensor to add, shape (1, 5)

    def forward(self, x):
        t1 = self.linear(x)  # Apply the linear transformation
        t2 = t1 + self.other_tensor  # Add another tensor
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initialize the model
model = CustomModel()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Input shape (1, 10)
output = model(input_tensor)

# Print output
print(output)
