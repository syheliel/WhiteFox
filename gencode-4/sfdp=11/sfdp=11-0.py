import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, inv_scale_factor):
        q = self.query_layer(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key_layer(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value_layer(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        t1 = torch.matmul(q, k.transpose(-2, -1))        # Perform matrix multiplication
        t2 = t1.div(inv_scale_factor)                     # Divide by inverse scale factor
        t3 = t2.softmax(dim=-1)                           # Apply softmax function
        t4 = nn.functional.dropout(t3, p=self.dropout_p) # Apply dropout
        output = t4.matmul(v)                             # Matrix multiplication with value tensor
        
        return output

# Initializing the model
embed_size = 64    # Size of the embedding
num_heads = 8      # Number of attention heads
dropout_p = 0.1    # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Inputs to the model
batch_size = 2
seq_length = 10
query_tensor = torch.randn(batch_size, seq_length, embed_size)  # Query tensor
key_tensor = torch.randn(batch_size, seq_length, embed_size)    # Key tensor
value_tensor = torch.randn(batch_size, seq_length, embed_size)  # Value tensor
inv_scale_factor = embed_size ** 0.5                       # Inverse scale factor for scaling

# Forward pass
output = model(query_tensor, key_tensor, value_tensor, inv_scale_factor)
print(output.shape)  # Output shape
