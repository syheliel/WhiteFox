import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.query_linear = torch.nn.Linear(query_dim, query_dim)
        self.key_linear = torch.nn.Linear(key_dim, key_dim)
        self.value_linear = torch.nn.Linear(value_dim, value_dim)

    def forward(self, query, key, value):
        # Transform the inputs with linear layers
        query_transformed = self.query_linear(query)
        key_transformed = self.key_linear(key)
        value_transformed = self.value_linear(value)

        # Compute attention weights
        attn_weight = torch.bmm(query_transformed, key_transformed.transpose(1, 2)).softmax(dim=-1)
        
        # Apply dropout to the attention weights
        attn_weight = torch.nn.functional.dropout(attn_weight, p=self.dropout_p)
        
        # Compute the output
        output = torch.bmm(attn_weight, value_transformed)
        return output

# Initializing the model with appropriate dimensions
query_dim = 64
key_dim = 64
value_dim = 64
dropout_p = 0.1
model = AttentionModel(query_dim, key_dim, value_dim, dropout_p)

# Inputs to the model
batch_size = 5
seq_length = 10
query = torch.randn(batch_size, seq_length, query_dim)
key = torch.randn(batch_size, seq_length, key_dim)
value = torch.randn(batch_size, seq_length, value_dim)

# Forward pass
output = model(query, key, value)
