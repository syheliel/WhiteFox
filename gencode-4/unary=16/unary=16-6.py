import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)  # Linear layer with input size 128 and output size 64

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.relu(t1)  # Apply ReLU activation
        return t2

# Initializing the model
model = Model()

# Generating an input tensor
input_tensor = torch.randn(1, 128)  # Batch size of 1 and input size of 128

# Forward pass through the model
output = model(input_tensor)

# Output the result
print(output)
