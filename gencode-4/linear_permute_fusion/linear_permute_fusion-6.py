import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer with input features of 10 and output features of 15
        self.linear = torch.nn.Linear(10, 15)
 
    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation to the input tensor
        # Permute the output tensor to swap the last two dimensions
        t2 = t1.permute(0, 2, 1)  # Assuming t1 has shape (batch_size, 3, 5) for permutation
        return t2

# Initializing the model
model = Model()

# Inputs to the model
# Create a random input tensor with shape (batch_size, channels, features)
# For the linear layer, we need to ensure the input dimension matches, so we set it to (1, 10)
x = torch.randn(1, 10)

# Get the output from the model
output = model(x)

print("Output shape:", output.shape)
