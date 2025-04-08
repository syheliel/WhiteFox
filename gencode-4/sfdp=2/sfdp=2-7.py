import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.scale_factor = embed_size ** 0.5  # Scaling factor for query-key dot product

        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk / self.scale_factor
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        # Apply dropout to the softmax output
        dropout_qk = nn.functional.dropout(softmax_qk, p=self.dropout_p)
        
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
        
        return output

# Initializing the model
embed_size = 64  # Size of the embedding
heads = 8  # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, heads, dropout_p)

# Inputs to the model
batch_size = 1  # Number of samples in a batch
seq_length = 10  # Length of the input sequence
query = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value = torch.randn(batch_size, seq_length, embed_size)  # Value tensor

# Forward pass
output = model(query, key, value)

# Output shape
print("Output shape:", output.shape)
