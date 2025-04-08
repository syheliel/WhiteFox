import torch

# Model Definition
class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)  # Linear transformation

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.relu(t1)  # Apply ReLU activation
        return t2

# Initializing the model
input_dim = 10  # Number of input features
output_dim = 5  # Number of output features
model = SimpleModel(input_dim, output_dim)

# Generating input tensor
input_tensor = torch.randn(1, input_dim)  # Batch size of 1, with input_dim features

# Forward pass
output = model(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("\nModel Output:")
print(output)
