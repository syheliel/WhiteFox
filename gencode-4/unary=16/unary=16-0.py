import torch

# Define the model
class LinearReLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a linear layer, input features = 128, output features = 64
        self.linear = torch.nn.Linear(128, 64)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.relu(t1)  # Apply the ReLU activation function
        return t2

# Initializing the model
model = LinearReLUModel()

# Generate an input tensor
input_tensor = torch.randn(1, 128)  # Batch size of 1 and 128 input features

# Run the model with the input tensor
output_tensor = model(input_tensor)

# Display the output tensor
print(output_tensor)
