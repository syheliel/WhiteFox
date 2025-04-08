import torch

class CustomModel(torch.nn.Module):
    def __init__(self, dropout_prob=0.2):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.fc = torch.nn.Linear(128, 64)  # A simple fully connected layer

    def forward(self, x):
        # Apply dropout to the input tensor
        t1 = torch.nn.functional.dropout(x, p=self.dropout_prob, training=self.training)
        # Generate a tensor with the same size as input_tensor filled with random numbers
        t2 = torch.rand_like(x)
        # Combine the results (for demonstration purposes)
        output = t1 + t2
        return output

# Initializing the model
model = CustomModel()

# Creating an input tensor
input_tensor = torch.randn(1, 128)  # Batch size of 1, 128 features
output_tensor = model(input_tensor)

# Output the result
print("Output Tensor Shape:", output_tensor.shape)
