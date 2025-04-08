import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale_factor = 1.0 / (embed_size ** 0.5)  # Scaling factor for attention
        self.dropout_p = dropout_p

    def forward(self, x):
        q = self.query(x)  # Compute query
        k = self.key(x)    # Compute key
        v = self.value(x)  # Compute value
        
        qk = torch.matmul(q, k.transpose(-2, -1))  # Compute dot product of query and key
        scaled_qk = qk.mul(self.scale_factor)  # Scale the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Apply softmax
        dropout_qk = F.dropout(softmax_qk, p=self.dropout_p, training=self.training)  # Apply dropout
        output = dropout_qk.matmul(v)  # Compute dot product of dropout output and value
        
        return output

# Initializing the model
embed_size = 64  # Size of each embedding
num_heads = 8    # Number of attention heads (not used directly in this example)
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generate an input tensor for the model
# For this example, we will create a batch of 10 sequences, each with 5 tokens, where each token is represented by an embedding of size 64
x_input = torch.randn(10, 5, embed_size)  # Shape: (batch_size, sequence_length, embed_size)

# Forward pass through the model
output = model(x_input)

print(output.shape)  # Output shape should be (10, 5, 64)
