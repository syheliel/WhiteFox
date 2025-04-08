import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)  # Linear layer transforming input of size 10 to output of size 1

    def forward(self, x):
        t1 = self.linear(x)             # Apply linear transformation
        t2 = torch.sigmoid(t1)          # Apply sigmoid function
        return t2

# Initializing the model
model = SimpleModel()

# Generate input tensor
input_tensor = torch.randn(1, 10)  # Input tensor with batch size of 1 and feature size of 10

# Forward pass through the model
output = model(input_tensor)

# Output for verification
print("Input Tensor:", input_tensor)
print("Output Tensor:", output)
