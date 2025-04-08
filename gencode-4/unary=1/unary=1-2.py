import torch

class CustomModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1  # Square the output
        t4 = t3 * t1  # Cube the output
        t5 = t4 * 0.044715  # Multiply by 0.044715
        t6 = t1 + t5  # Add the linear output to the cubic output
        t7 = t6 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)  # Apply tanh
        t9 = t8 + 1  # Add 1
        t10 = t2 * t9  # Multiply t2 by t9
        return t10

# Initialize the model with input dimension 10 and output dimension 5
model = CustomModel(input_dim=10, output_dim=5)

# Input tensor for the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input dimension of 10

# Get the output from the model
output = model(input_tensor)

print(output)
