import torch
import torch.nn as nn

device = 'cpu'

emb = nn.Embedding(10, 12).to(device)
inp1 = torch.LongTensor([1, 2, 3, 4]).to(device)
inp1 = emb(inp1).reshape(inp1.shape[0], 1, 12) #S N E

encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=4, dropout=0).to(device)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4).to(device)

out1 = transformer_encoder(inp1)
out2 = transformer_encoder(inp1)

print((out1-out2).norm())