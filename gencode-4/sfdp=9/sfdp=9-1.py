import torch
import math

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttentionModel, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Linear layers for query, key, and value
        self.query_layer = torch.nn.Linear(d_model, d_model)
        self.key_layer = torch.nn.Linear(d_model, d_model)
        self.value_layer = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # Permute the dimensions of the query tensor
        q = self.query_layer(query).permute(0, 2, 1, 3)  # Assuming input shape [batch_size, seq_len, num_heads, d_k]
        k = self.key_layer(key).permute(0, 2, 1, 3)      # Same for key
        v = self.value_layer(value).permute(0, 2, 1, 3)  # Same for value
        
        # Scale the query tensor
        q = q / math.sqrt(q.size(-1))
        
        # Matrix multiplication between query and transposed key
        div = q @ k.transpose(-2, -1)
        
        # Convert to float32
        div = div.to(torch.float32)
        
        # Apply softmax to get attention weights
        attn_weight = torch.softmax(div, dim=-1)
        
        # Convert attention weights to float16
        attn_weight = attn_weight.to(torch.float16)
        
        # Matrix multiplication between attention weights and value tensor
        output = attn_weight @ v
        return output

# Initializing the model
d_model = 64  # Size of the embedding
num_heads = 8  # Number of attention heads
model = SelfAttentionModel(d_model, num_heads)

# Inputs to the model
batch_size = 1
seq_len = 10  # Sequence length
input_tensor = torch.randn(batch_size, seq_len, d_model)  # Random input tensor for query, key, and value

# Since query, key, and value are expected to have the same shape
output = model(input_tensor, input_tensor, input_tensor)

# Print output shape
print(output.shape)  # Should be [batch_size, num_heads, seq_len, d_k]
