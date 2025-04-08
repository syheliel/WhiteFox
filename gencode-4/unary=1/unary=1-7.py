import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Input features: 10, Output features: 5
 
    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 * 0.5  # Multiply the output of the linear transformation by 0.5
        t3 = t1 * t1  # Square the output of the linear transformation
        t4 = t3 * t1  # Cube the output of the linear transformation
        t5 = t4 * 0.044715  # Multiply the output of the cube by 0.044715
        t6 = t1 + t5  # Add the output of the linear transformation to the output of the multiplication
        t7 = t6 * 0.7978845608028654  # Multiply the output of the addition by 0.7978845608028654
        t8 = torch.tanh(t7)  # Apply the hyperbolic tangent function
        t9 = t8 + 1  # Add 1
        t10 = t2 * t9  # Multiply the output of the linear transformation by the output of the hyperbolic tangent function
        return t10

# Initialize the model
model = CustomModel()

# Generate input tensor for the model
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input features of 10

# Get the output from the model
output = model(input_tensor)

# Print output
print(output)
