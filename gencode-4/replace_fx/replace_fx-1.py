import torch

class CustomModel(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.linear = torch.nn.Linear(10, 20)  # A linear layer for demonstration
    
    def forward(self, x):
        # Apply dropout to the input tensor
        t1 = torch.nn.functional.dropout(x, p=self.dropout_prob)
        # Generate a tensor with the same size as input_tensor filled with random numbers
        t2 = torch.rand_like(x)
        
        # Some arbitrary operations to combine t1 and t2
        output = t1 + t2
        return output

# Initializing the model
model = CustomModel()

# Input tensor for the model
input_tensor = torch.randn(5, 10)  # Batch size of 5, feature size of 10
output = model(input_tensor)

# Displaying the output
print(output)
