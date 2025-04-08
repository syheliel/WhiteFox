import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features
    
    def forward(self, input_tensor, other):
        t1 = self.linear(input_tensor)  # Apply linear transformation
        t2 = t1 + other  # Add another tensor to the output of the linear transformation
        return t2

# Initializing the model
model = Model()

# Generating input tensors
input_tensor = torch.randn(1, 10)  # Batch size of 1 with 10 input features
other = torch.randn(1, 5)  # Tensor to add with the same shape as the output of the linear layer

# Forward pass through the model
output = model(input_tensor, other)

print("Input Tensor:", input_tensor)
print("Other Tensor:", other)
print("Output Tensor:", output)
