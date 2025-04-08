import torch

# Define the model class
class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, inv_scale_factor):
        q = self.query(query).permute(0, 2, 1)  # Permute dimensions for query
        k = self.key(key).permute(0, 2, 1)      # Permute dimensions for key
        v = self.value(value).permute(0, 2, 1)  # Permute dimensions for value
        
        t1 = torch.matmul(q, k.transpose(-2, -1))  # Matrix multiplication
        t2 = t1.div(inv_scale_factor)                # Divide by inverse scale factor
        t3 = t2.softmax(dim=-1)                      # Apply softmax
        t4 = torch.nn.functional.dropout(t3, p=self.dropout_p)  # Apply dropout
        output = t4.matmul(v)                        # Final matrix multiplication
        
        return output

# Initialize the model
embed_dim = 64  # Example embedding dimension
num_heads = 8   # Number of attention heads (not used in this simple model)
dropout_p = 0.1 # Dropout probability
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
seq_length = 10  # Example sequence length
query_tensor = torch.randn(batch_size, seq_length, embed_dim)  # Query tensor
key_tensor = torch.randn(batch_size, seq_length, embed_dim)    # Key tensor
value_tensor = torch.randn(batch_size, seq_length, embed_dim)  # Value tensor
inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor (sqrt of the embedding dimension)

# Get output from the model
output = model(query_tensor, key_tensor, value_tensor, inv_scale_factor)

print(output.shape)  # Output shape
