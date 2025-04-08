import torch

# Model definition
class AdditiveModel(torch.nn.Module):
    def __init__(self, input_size, output_size, other_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.other = torch.randn(1, other_size)  # Predefined tensor for addition

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = t1 + self.other  # Add another tensor to the output
        return t2

# Initializing the model
input_size = 10  # Size of the input features
output_size = 5  # Size of the output features after linear transformation
other_size = output_size  # Size of the tensor to be added
model = AdditiveModel(input_size, output_size, other_size)

# Creating an input tensor
x = torch.randn(1, input_size)  # A batch size of 1 with input_size features

# Forward pass through the model
output = model(x)

# Displaying the output
print(output)
