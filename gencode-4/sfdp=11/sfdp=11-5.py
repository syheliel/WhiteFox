import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.query_layer = torch.nn.Linear(embed_size, embed_size)
        self.key_layer = torch.nn.Linear(embed_size, embed_size)
        self.value_layer = torch.nn.Linear(embed_size, embed_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_size ** 0.5  # Inverse scale factor for scaling

    def forward(self, query, key, value):
        q = self.query_layer(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key_layer(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value_layer(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        t1 = torch.matmul(q, k.transpose(-2, -1))       # Matrix multiplication between query and transposed key
        t2 = t1.div(self.inv_scale_factor)                # Divide by the inverse scale factor
        t3 = t2.softmax(dim=-1)                           # Apply softmax
        t4 = self.dropout(t3)                             # Apply dropout
        output = t4.matmul(v)                             # Matrix multiplication with value tensor
        
        return output

# Initializing the model
embed_size = 64  # Size of the embedding
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Generating random input tensors
batch_size = 2  # Number of samples in a batch
seq_length = 10 # Length of the input sequence

query_tensor = torch.randn(batch_size, seq_length, embed_size)
key_tensor = torch.randn(batch_size, seq_length, embed_size)
value_tensor = torch.randn(batch_size, seq_length, embed_size)

# Forward pass through the model
output = model(query_tensor, key_tensor, value_tensor)

# Output shape
print("Output shape:", output.shape)  # Should be (batch_size, seq_length, embed_size)
