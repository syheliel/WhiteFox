import torch

class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 10)  # Linear layer with input size 20 and output size 10
        self.other = torch.randn(1, 10)  # A tensor to be added (same size as linear output)

    def forward(self, input_tensor):
        t1 = self.linear(input_tensor)  # Apply the linear transformation
        t2 = t1 + self.other  # Add another tensor
        t3 = torch.relu(t2)  # Apply the ReLU activation function
        return t3

# Initializing the model
model = CustomModel()

# Creating an input tensor for the model (batch size of 1 and input size of 20)
input_tensor = torch.randn(1, 20)

# Forward pass through the model
output = model(input_tensor)

# Display the output
print(output)
