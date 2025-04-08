import torch

class CustomModel(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.fc1 = torch.nn.Linear(256, 128)  # Fully connected layer

    def forward(self, x):
        # Apply dropout to the input tensor
        t1 = torch.nn.functional.dropout(x, p=self.dropout_prob, training=self.training)  
        
        # Generate a tensor with the same size as input_tensor filled with random numbers
        t2 = torch.rand_like(x)
        
        # Pass through the fully connected layer
        output = self.fc1(t1 + t2)  # Combine t1 and t2 before passing to the linear layer
        return output

# Initializing the model
model = CustomModel()

# Inputs to the model
input_tensor = torch.randn(1, 256)  # Batch size of 1 and 256 features
output = model(input_tensor)

# Printing the output tensor
print(output)
