import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear layer with input size 10 and output size 5

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.relu(t1)  # Apply ReLU activation
        return t2

# Initializing the model
model = SimpleModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10
output = model(input_tensor)

# Print the output shape
print("Output shape:", output.shape)
