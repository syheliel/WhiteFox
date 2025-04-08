import torch

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)  # Linear transformation from 128 to 64 dimensions

    def forward(self, x):
        t1 = self.linear(x)  # Apply linear transformation
        t2 = torch.tanh(t1)  # Apply the hyperbolic tangent function
        return t2

# Initializing the model
model = Model()

# Input tensor: A random tensor with shape (batch_size, number_of_features)
input_tensor = torch.randn(32, 128)  # Batch size of 32 and 128 features

# Forward pass to get the output
output = model(input_tensor)

# Display the output shape
print(output.shape)  # Should be (32, 64)
