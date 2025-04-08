import torch

class LinearSigmoidModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        t1 = self.linear(x)  # Apply a linear transformation
        t2 = torch.sigmoid(t1)  # Apply the sigmoid function
        return t2

# Initializing the model
input_dim = 10  # Example input dimension
output_dim = 1  # Example output dimension (for binary classification)
model = LinearSigmoidModel(input_dim, output_dim)

# Generating an input tensor
input_tensor = torch.randn(1, input_dim)  # Batch size of 1
output = model(input_tensor)

print("Input tensor:", input_tensor)
print("Output tensor:", output)
