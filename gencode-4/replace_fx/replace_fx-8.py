import torch

# Define the model
class DropoutRandLikeModel(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.linear = torch.nn.Linear(10, 20)  # A simple linear layer

    def forward(self, x):
        # Apply dropout to the input tensor
        t1 = torch.nn.functional.dropout(x, p=self.dropout_prob)
        # Generate a tensor with the same size as input_tensor filled with random numbers
        t2 = torch.rand_like(x)
        # Combine the results in some way (for demonstration)
        return t1 + t2

# Initializing the model
model = DropoutRandLikeModel()

# Inputs to the model
input_tensor = torch.randn(1, 10)  # Batch size of 1, input size of 10
output = model(input_tensor)

# Print the output
print(output)
