import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(10, 20)

    def forward(self, tensor1, tensor2):
        # Concatenate tensors along dimension 1
        t1 = torch.cat([tensor1, tensor2], dim=1)
        # Reshape the concatenated tensor
        t2 = t1.view(-1, 20)  # Assuming the concatenated result is of suitable size
        # Apply ReLU activation
        t3 = torch.relu(t2)
        return t3

# Initializing the model
model = MyModel()

# Creating input tensors
tensor1 = torch.randn(5, 10)  # Batch of 5 with 10 features
tensor2 = torch.randn(5, 10)  # Batch of 5 with 10 features

# Forward pass
output = model(tensor1, tensor2)

# Print the output shape
print(output.shape)
