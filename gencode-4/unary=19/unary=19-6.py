import torch

# Model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)  # Linear transformation from 10 to 5 dimensions

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation to the input tensor
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function to the output of the linear transformation
        return t2

# Initializing the model
model = SimpleModel()

# Input tensor to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, 10 features

# Getting the output from the model
output = model(input_tensor)

# Print the output
print(output)
