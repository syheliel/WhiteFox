import torch

class AttentionModel(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.inv_scale_factor = embed_dim ** 0.5  # Inverse scale factor for attention

    def forward(self, x):
        q = self.query(x).permute(0, 2, 1)  # Permute dimensions of query tensor
        k = self.key(x).permute(0, 2, 1)    # Permute dimensions of key tensor
        v = self.value(x).permute(0, 2, 1)  # Permute dimensions of value tensor

        t1 = torch.matmul(q, k.transpose(-2, -1))  # Matrix multiplication
        t2 = t1 / self.inv_scale_factor               # Divide by inverse scale factor
        t3 = t2.softmax(dim=-1)                       # Apply softmax
        t4 = self.dropout(t3)                         # Apply dropout
        output = t4.matmul(v)                         # Matrix multiplication with value tensor
        
        return output

# Parameters
embed_dim = 64  # Size of the embeddings
num_heads = 8   # Number of attention heads (not used in this simplified version)
dropout_p = 0.1 # Dropout probability

# Initialize the model
model = AttentionModel(embed_dim=embed_dim, num_heads=num_heads, dropout_p=dropout_p)

# Sample input tensor (batch_size=1, seq_length=10, embed_dim=64)
x_input = torch.randn(1, 10, embed_dim)

# Get the output from the model
output = model(x_input)

print("Output shape:", output.shape)  # Should be (1, 10, embed_dim)
