import torch

# Define the Model class
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout_p):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, inv_scale_factor):
        q = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = key.permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = value.permute(0, 2, 1, 3)    # Permute the dimensions of the value tensor
        
        t1 = torch.matmul(q, k.transpose(-2, -1))  # Matrix multiplication
        t2 = t1.div(inv_scale_factor)                # Divide by inverse scale factor
        t3 = t2.softmax(dim=-1)                      # Apply softmax
        t4 = self.dropout(t3)                        # Apply dropout
        output = t4.matmul(v)                        # Matrix multiplication with value tensor
        
        return output

# Initialize the model
embed_size = 64  # Size of embeddings
num_heads = 8    # Number of attention heads
dropout_p = 0.1  # Dropout probability
model = AttentionModel(embed_size, num_heads, dropout_p)

# Create input tensors
batch_size = 2
seq_length = 10
num_features = embed_size

# Example input tensors for query, key, and value
query = torch.randn(batch_size, seq_length, num_heads, num_features // num_heads)
key = torch.randn(batch_size, seq_length, num_heads, num_features // num_heads)
value = torch.randn(batch_size, seq_length, num_heads, num_features // num_heads)

# Define the inverse scale factor
inv_scale_factor = torch.sqrt(torch.tensor(num_features // num_heads, dtype=torch.float32))

# Forward pass through the model
output = model(query, key, value, inv_scale_factor)

# Output tensor
print(output.shape)  # Should be of shape (batch_size, seq_length, num_heads, num_features // num_heads)
