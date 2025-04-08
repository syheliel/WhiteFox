import torch

# Model Definition
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(128, 64)  # Linear transformation from 128 to 64 dimensions
    
    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.relu(t1)  # Apply the ReLU activation function
        return t2

# Initializing the model
model = Model()

# Generating an input tensor
# The input should match the input dimensions of the Linear layer (128 in this case)
input_tensor = torch.randn(1, 128)  # Batch size of 1 and input feature size of 128

# Getting the output from the model
output = model(input_tensor)

# Printing the output shape
print("Output shape:", output.shape)
