import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionModel(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(SelfAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Layers for query, key, and value transformations
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        k = self.key_layer(key).permute(0, 2, 1, 3)      # (batch_size, seq_len, num_heads, head_dim)
        v = self.value_layer(value).permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        
        # Scale the query tensor
        q = q / math.sqrt(q.size(-1))
        
        # Matrix multiplication between query and transposed key tensor
        div = q @ k.transpose(-2, -1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Convert to float32
        div = div.to(torch.float32)
        
        # Apply softmax to the result along the last dimension
        attn_weight = F.softmax(div, dim=-1)
        
        # Apply dropout to the softmax result
        attn_weight = self.dropout(attn_weight)
        
        # Convert to float16
        attn_weight = attn_weight.to(torch.float16)

        # Matrix multiplication between the result and the value tensor
        output = attn_weight @ v  # (batch_size, num_heads, seq_len, head_dim)
        
        return output

# Initialize the model
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Create a sample input tensor for the model
batch_size = 1
seq_len = 10  # Length of the sequence
input_tensor = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

# Call the model with input tensors
output = model(input_tensor, input_tensor, input_tensor)

print(output.shape)  # Output shape
