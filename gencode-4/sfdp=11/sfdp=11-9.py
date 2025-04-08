import torch

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, embed_dim, dropout_p=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale_factor):
        # Permute the dimensions of the query, key, and value tensors
        q = query.permute(0, 2, 1, 3)  # (batch_size, seq_len, heads, embed_dim)
        k = key.permute(0, 2, 1, 3)    # (batch_size, seq_len, heads, embed_dim)
        v = value.permute(0, 2, 1, 3)  # (batch_size, seq_len, heads, embed_dim)
        
        # Perform matrix multiplication between the query tensor and the transposed key tensor
        t1 = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, heads, seq_len, seq_len)
        
        # Divide the result by an inverse scale factor
        t2 = t1.div(inv_scale_factor)  # (batch_size, heads, seq_len, seq_len)
        
        # Apply softmax function to the result
        t3 = t2.softmax(dim=-1)  # (batch_size, heads, seq_len, seq_len)
        
        # Apply dropout to the result
        t4 = self.dropout(t3)  # (batch_size, heads, seq_len, seq_len)
        
        # Perform matrix multiplication between the result and the value tensor
        output = t4.matmul(v)  # (batch_size, heads, seq_len, embed_dim)

        return output

# Example usage
embed_dim = 64
seq_len = 10
batch_size = 2
inv_scale_factor = embed_dim ** 0.5  # Typically, the inverse scaling factor for attention
dropout_p = 0.1

# Initialize the model
attention_model = ScaledDotProductAttention(embed_dim, dropout_p)

# Create random input tensors for query, key, and value
query = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)    # (batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

# Forward pass through the model
output = attention_model(query, key, value, inv_scale_factor)

print("Output shape:", output.shape)  # Should be (batch_size, seq_len, embed_dim)
