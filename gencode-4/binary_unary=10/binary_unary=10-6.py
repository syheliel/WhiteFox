import torch

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 15)  # Linear layer with input size 10 and output size 15
        self.other = torch.randn(1, 15)  # A tensor to add, with the same size as the output of the linear layer

    def forward(self, x):
        t1 = self.linear(x)        # Apply a linear transformation to the input tensor
        t2 = t1 + self.other       # Add another tensor to the output of the linear transformation
        t3 = torch.relu(t2)        # Apply the ReLU activation function to the result
        return t3

# Initializing the model
model = LinearModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Example input tensor with size (1, 10)
output = model(input_tensor)  # Get the output from the model
