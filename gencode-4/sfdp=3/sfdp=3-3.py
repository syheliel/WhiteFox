import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p=0.1):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.scale_factor = 1.0 / (embed_size ** 0.5)  # Scaling factor for attention scores
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        # Compute dot product of query and key
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product
        scaled_qk = qk.mul(self.scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = F.dropout(softmax_qk, p=self.dropout_p, training=self.training)
        
        # Compute the dot product of the dropout output and value
        output = dropout_qk.matmul(value)
        
        return output

# Initialize the model
embed_size = 64  # Example embedding size
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Create example input tensors
batch_size = 1
seq_length = 10
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query, key, value)

# Output shape
print("Output shape:", output.shape)

batch_size = 1
seq_length = 10
embed_size = 64
query = torch.randn(batch_size, seq_length, embed_size)
key = torch.randn(batch_size, seq_length, embed_size)
value = torch.randn(batch_size, seq_length, embed_size)
