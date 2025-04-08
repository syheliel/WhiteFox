import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionModel(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Query, Key, Value Linear layers
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value):
        # Permute dimensions for multi-head attention
        q = self.query_layer(query).permute(0, 2, 1, 3)
        k = self.key_layer(key).permute(0, 2, 1, 3)
        v = self.value_layer(value).permute(0, 2, 1, 3)
        
        # Compute scaled dot-product attention
        div = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        div = div.to(torch.float32)  # Convert to float32
        attn_weight = F.softmax(div, dim=-1)  # Apply softmax
        attn_weight = self.dropout(attn_weight)  # Apply dropout
        attn_weight = attn_weight.to(torch.float16)  # Convert to float16
        
        # Compute the attention output
        output = attn_weight @ v
        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generating input tensors
batch_size = 1
seq_length = 10  # Length of the sequence
input_tensor = torch.randn(batch_size, seq_length, embed_size)  # Random input tensor

# Example of using the model
output = model(input_tensor, input_tensor, input_tensor)
print(output.shape)  # Should output the shape of the attention output
