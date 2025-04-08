import torch

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define a linear layer with input size 10 and output size 5
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        t1 = self.linear(x)      # Apply the linear transformation
        t2 = torch.tanh(t1)      # Apply the hyperbolic tangent function
        return t2

# Initialize the model
model = Model()

# Generate an input tensor with a batch size of 1 and input size of 10
input_tensor = torch.randn(1, 10)

# Pass the input tensor through the model
output_tensor = model(input_tensor)

# Print the output
print("Output tensor:", output_tensor)
