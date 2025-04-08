import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1  # Square the output
        t4 = t3 * t1  # Cube the output
        t5 = t4 * 0.044715  # Multiply by 0.044715
        t6 = t1 + t5  # Add linear output to the multiplication
        t7 = t6 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)  # Apply hyperbolic tangent function
        t9 = t8 + 1  # Add 1
        t10 = t2 * t9  # Multiply by the output of the hyperbolic tangent function
        return t10

# Initialize the model
model = Model()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Create a random input tensor with shape (1, 10)

# Get the model output
output = model(input_tensor)

# Display the output
print(output)
