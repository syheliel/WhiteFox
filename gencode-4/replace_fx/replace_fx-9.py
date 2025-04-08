import torch

# Define the model
class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 64)  # A fully connected layer

    def forward(self, x):
        t1 = torch.nn.functional.dropout(x, p=0.5)  # Apply dropout to the input tensor
        t2 = torch.rand_like(x)  # Generate a tensor with the same size as input_tensor filled with random numbers
        out = self.fc(t1 + t2)  # Add dropout output and random tensor and pass through the linear layer
        return out

# Initializing the model
model = CustomModel()

# Create an input tensor for the model
input_tensor = torch.randn(1, 128)  # Batch size of 1 and 128 features

# Get the output from the model
output = model(input_tensor)

# Printing the output shape
print(output.shape)
