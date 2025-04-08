import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.1):
        super(AttentionModel, self).__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        # Assuming x has shape (batch_size, seq_length, embed_dim)
        q = self.query(x)  # Shape: (batch_size, seq_length, embed_dim)
        k = self.key(x)    # Shape: (batch_size, seq_length, embed_dim)
        v = self.value(x)  # Shape: (batch_size, seq_length, embed_dim)
        
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(q, k.transpose(-2, -1))  # Shape: (batch_size, seq_length, seq_length)
        
        # Scale the dot product by an inverse scale factor
        inv_scale_factor = k.size(-1) ** 0.5  # Typically sqrt(d_k)
        scaled_qk = qk.div(inv_scale_factor)  # Shape: (batch_size, seq_length, seq_length)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)  # Shape: (batch_size, seq_length, seq_length)
        
        # Apply dropout to the softmax output
        dropout_qk = self.dropout(softmax_qk)  # Shape: (batch_size, seq_length, seq_length)
        
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(v)  # Shape: (batch_size, seq_length, embed_dim)
        
        return output

# Initializing the model
embed_dim = 64
num_heads = 8
dropout_p = 0.1
model = AttentionModel(embed_dim, num_heads, dropout_p)

# Input tensor
batch_size = 1
seq_length = 10
input_tensor = torch.randn(batch_size, seq_length, embed_dim)

# Forward pass
output = model(input_tensor)
print(output.shape)  # Should output: torch.Size([1, 10, 64])
