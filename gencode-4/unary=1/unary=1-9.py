import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)  # Linear transformation with input and output size of 10

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 * 0.5  # Multiply by 0.5
        t3 = t1 * t1  # Square the output
        t4 = t3 * t1  # Cube the output
        t5 = t4 * 0.044715  # Multiply the cube by 0.044715
        t6 = t1 + t5  # Add the linear output to the multiplication
        t7 = t6 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)  # Apply hyperbolic tangent
        t9 = t8 + 1  # Add 1
        t10 = t2 * t9  # Multiply by the earlier result
        return t10

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Example input tensor of shape (batch_size, input_features)

# Forward pass
output = model(input_tensor)

# Displaying the output
print(output)
