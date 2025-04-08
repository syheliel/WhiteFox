import torch

class DropoutRandLikeModel(torch.nn.Module):
    def __init__(self, dropout_probability=0.5):
        super().__init__()
        self.dropout_probability = dropout_probability

    def forward(self, input_tensor):
        # Apply dropout to the input tensor
        t1 = torch.nn.functional.dropout(input_tensor, p=self.dropout_probability, training=self.training)
        
        # Generate a tensor with the same size as input_tensor filled with random numbers
        t2 = torch.rand_like(input_tensor)
        
        # Example operation (just to illustrate the flow)
        output_tensor = t1 + t2  # Combine the results in some way
        return output_tensor

# Initializing the model
model = DropoutRandLikeModel()

# Inputs to the model
input_tensor = torch.randn(1, 3, 64, 64)
output = model(input_tensor)

print(output)
