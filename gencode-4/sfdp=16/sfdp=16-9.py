import torch

class SelfAttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        # Define the query, key, and value linear layers
        self.query_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.key_layer = torch.nn.Linear(embed_dim, embed_dim)
        self.value_layer = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        # Permute the dimensions of the query, key, and value tensors
        q = self.query_layer(query).permute(0, 2, 1, 3)
        k = self.key_layer(key).permute(0, 2, 1, 3)
        v = self.value_layer(value).permute(0, 2, 1, 3)

        bs = q.size(0)  # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)  # Get the size of the second to last dimension of the key tensor
        
        # Compute the dot product of the query tensor and the transposed key tensor
        scores = q @ k.transpose(-2, -1)
        scores = scores.div(inv_scale)  # Divide the scores by the inverse scale
        
        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Create a tensor filled with negative infinity
        
        if attn_mask is not None:
            attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Create an attention mask
            scores = scores.masked_fill(attn_mask, fill_value)  # Mask the scores
        
        # Apply dropout to the softmax of the scores
        attn_weights = torch.nn.functional.dropout(torch.softmax(scores, dim=-1), p=self.dropout_p)
        output = attn_weights @ v  # Compute the dot product with the value tensor
        
        return output

# Initializing the model
embed_dim = 64  # Embedding dimension
num_heads = 8   # Number of attention heads
dropout_p = 0.1 # Dropout probability
model = SelfAttentionModel(embed_dim, num_heads, dropout_p)

# Input tensors for the model
batch_size = 2
seq_length = 10
input_dim = embed_dim  # The input dimension should match the embedding dimension

query_tensor = torch.randn(batch_size, seq_length, input_dim)
key_tensor = torch.randn(batch_size, seq_length, input_dim)
value_tensor = torch.randn(batch_size, seq_length, input_dim)
attn_mask_tensor = torch.zeros(batch_size, seq_length)  # Attention mask (0 indicates attend)

# Getting the output from the model
output = model(query_tensor, key_tensor, value_tensor, attn_mask_tensor)
print(output.shape)  # Should print (batch_size, seq_length, num_heads, embed_dim // num_heads)
