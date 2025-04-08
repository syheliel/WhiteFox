import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 input features to 5 output features

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = t1 * 0.5  # Multiply the output by 0.5
        t3 = t1 * t1  # Square the output
        t4 = t3 * t1  # Cube the output
        t5 = t4 * 0.044715  # Multiply the cube by 0.044715
        t6 = t1 + t5  # Add the linear output to the multiplication output
        t7 = t6 * 0.7978845608028654  # Multiply by 0.7978845608028654
        t8 = torch.tanh(t7)  # Apply the hyperbolic tangent function
        t9 = t8 + 1  # Add 1 to the hyperbolic tangent output
        t10 = t2 * t9  # Multiply the output of the linear transformation by the hyperbolic tangent output
        return t10

# Initializing the model
model = Model()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Example input tensor with batch size 1 and 10 features
output = model(input_tensor)

print(output)
