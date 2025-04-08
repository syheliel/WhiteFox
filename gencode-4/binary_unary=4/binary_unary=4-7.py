import torch

class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, input_tensor, other):
        t1 = self.linear(input_tensor)  # Apply a linear transformation
        t2 = t1 + other                  # Add another tensor
        t3 = torch.nn.functional.relu(t2)  # Apply the ReLU activation function
        return t3

# Initializing the model with input and output dimensions
input_dim = 10  # Example input dimension
output_dim = 5  # Example output dimension
model = Model(input_dim, output_dim)

# Generating input tensors
input_tensor = torch.randn(1, input_dim)  # Batch size of 1 and input_dim features
other = torch.randn(1, output_dim)         # Tensor to add to the output

# Getting the model output
output = model(input_tensor, other)

print("Input Tensor:", input_tensor)
print("Other Tensor:", other)
print("Model Output:", output)
