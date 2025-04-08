import torch

# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)  # Linear transformation from 10 input features to 1 output

    def forward(self, x):
        t1 = self.linear(x)          # Apply linear transformation
        t2 = torch.sigmoid(t1)      # Apply sigmoid function
        return t2

# Initializing the model
model = SimpleModel()

# Input tensor to the model
input_tensor = torch.randn(5, 10)  # Batch size of 5, 10 features

# Forward pass
output = model(input_tensor)

print("Input Tensor:\n", input_tensor)
print("Model Output:\n", output)
