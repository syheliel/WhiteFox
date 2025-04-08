import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, input_tensor, other):
        t1 = self.linear(input_tensor)  # Apply a linear transformation
        t2 = t1 + other  # Add another tensor to the output
        t3 = torch.relu(t2)  # Apply ReLU activation function
        return t3

# Initializing the model
input_size = 10  # Example input size
output_size = 5  # Example output size
model = Model(input_size, output_size)

# Inputs to the model
input_tensor = torch.randn(1, input_size)  # Batch size of 1
other = torch.randn(1, output_size)  # Must match the output size of the linear layer

# Forward pass
output = model(input_tensor, other)

# Display the output
print(output)
