import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by an inverse scale factor
        inv_scale_factor = self.embed_size ** 0.5
        scaled_qk = qk.div(inv_scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = F.dropout(softmax_qk, p=self.dropout_p, training=self.training)
        
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
        return output

# Initializing the model parameters
embed_size = 64  # Size of the embedding
num_heads = 8  # Number of attention heads
dropout_p = 0.1  # Dropout probability

# Creating an instance of the model
attention_model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 1  # Number of samples in a batch
seq_length = 10  # Length of the sequence
query = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_size)  # Value tensor

# Forward pass through the model
output = attention_model(query, key, value)
