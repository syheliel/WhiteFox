import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(AttentionModel, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask, inv_scale):
        q = self.query_linear(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.key_linear(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.value_linear(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor
        
        attn_weights = (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1)  # Compute the attention weights
        dropout_attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p)  # Apply dropout to the attention weights
        output = dropout_attn_weights.matmul(v)  # Multiply the dropout attention weights by the value tensor
        
        return output

# Example initialization
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Inputs to the model
batch_size = 1
sequence_length = 10
num_value_heads = embed_dim // num_heads  # Assuming the embed_dim is divisible by num_heads
query_tensor = torch.randn(batch_size, sequence_length, embed_dim)
key_tensor = torch.randn(batch_size, sequence_length, embed_dim)
value_tensor = torch.randn(batch_size, sequence_length, embed_dim)
attn_mask = torch.zeros(batch_size, 1, sequence_length, sequence_length)  # Example attention mask
inv_scale = embed_dim ** -0.5  # Inverse scale for attention

# Forward pass
output = model(query_tensor, key_tensor, value_tensor, attn_mask, inv_scale)
print(output.shape)
