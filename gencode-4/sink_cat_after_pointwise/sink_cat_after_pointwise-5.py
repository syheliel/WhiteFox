import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a linear layer for demonstration purposes
        self.linear = torch.nn.Linear(10, 5)  # Input of size 10, output of size 5

    def forward(self, tensor1, tensor2):
        # Concatenate the two input tensors along dimension 1
        t1 = torch.cat((tensor1, tensor2), dim=1)
        
        # Reshape the concatenated tensor
        t2 = t1.view(-1, 5, 2)  # Assuming the concatenated size allows this shape
        
        # Apply the ReLU activation function to the reshaped tensor
        t3 = torch.relu(t2)
        
        return t3

# Initialize the model
model = MyModel()

# Generate input tensors
tensor1 = torch.randn(4, 5)  # 4 samples, 5 features
tensor2 = torch.randn(4, 5)  # 4 samples, 5 features

# Run the model with the input tensors
output = model(tensor1, tensor2)

# Display the output
print(output)
