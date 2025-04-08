import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x1):
        t1 = self.linear(x1)  # Apply linear transformation
        t2 = t1 * 0.5  # Multiply the output by 0.5
        t3 = t1 * t1  # Square the output
        t4 = t3 * t1  # Cube the output
        t5 = t4 * 0.044715  # Multiply by 0.044715
        t6 = t1 + t5  # Add the linear output and the scaled cube
        t7 = t6 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)  # Apply hyperbolic tangent
        t9 = t8 + 1  # Add 1
        t10 = t2 * t9  # Multiply by the output of the linear transformation scaled by 0.5
        return t10

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input features of size 10
output = model(input_tensor)

# Display the output
print(output)
