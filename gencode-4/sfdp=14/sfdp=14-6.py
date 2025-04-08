import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionModel, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.values = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.keys = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.queries = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = torch.nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, attn_mask=None, inv_scale=1.0):
        q = self.queries(query).permute(0, 2, 1, 3)  # Permute the dimensions of the query tensor
        k = self.keys(key).permute(0, 2, 1, 3)      # Permute the dimensions of the key tensor
        v = self.values(value).permute(0, 2, 1, 3)  # Permute the dimensions of the value tensor

        bs = q.size(0)  # Get the size of the first dimension of the query tensor
        k_len = k.size(-2)  # Get the size of the second to last dimension of the key tensor
        scores = q @ k.transpose(-2, -1)  # Compute the dot product of the query and the transposed key tensor
        scores = scores.div(inv_scale)  # Divide the scores by the inverse scale

        fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)  # Create a tensor filled with negative infinity
        attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)  # Create an attention mask and expand it to the size of the scores tensor
        attn_weights = torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1)  # Apply the softmax function to the scores tensor, fill the masked values with negative infinity
        
        out = attn_weights @ v  # Compute the dot product with the value tensor
        return self.fc_out(out.permute(0, 2, 1, 3))  # Permute back and pass through fully connected layer

# Initializing the model
embed_size = 64  # Size of each embedding
heads = 8  # Number of attention heads
model = AttentionModel(embed_size, heads)

# Inputs to the model
batch_size = 2
sequence_length = 10
query = torch.randn(batch_size, sequence_length, embed_size)
key = torch.randn(batch_size, sequence_length, embed_size)
value = torch.randn(batch_size, sequence_length, embed_size)
attn_mask = torch.zeros(batch_size, sequence_length)  # Example attention mask where all positions are valid

# Forward pass
output = model(query, key, value, attn_mask)
