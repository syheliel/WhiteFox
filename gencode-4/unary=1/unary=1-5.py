import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a linear layer with input size 10 and output size 5
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, input_tensor):
        t1 = self.linear(input_tensor)  # Apply linear transformation
        t2 = t1 * 0.5                   # Multiply the output by 0.5
        t3 = t1 * t1                    # Square the output
        t4 = t3 * t1                     # Cube the output
        t5 = t4 * 0.044715              # Multiply the cube by 0.044715
        t6 = t1 + t5                    # Add the linear output to the result
        t7 = t6 * 0.7978845608028654    # Multiply by constant
        t8 = torch.tanh(t7)             # Apply tanh
        t9 = t8 + 1                     # Add 1
        t10 = t2 * t9                   # Final multiplication
        return t10

# Initialize the model
model = MyModel()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Batch size of 1 and input size of 10

# Pass the input through the model
output = model(input_tensor)

# Display the output
print(output)

input_tensor = torch.randn(1, 10)  # Batch size of 1 with 10 features
