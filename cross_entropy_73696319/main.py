import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

batch_size = 4
seq_len = 8
vocab_size = 3

inputs = torch.randn((batch_size, vocab_size, seq_len), requires_grad=True)
target = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

loss1 = loss(inputs, target)
grad1 = torch.autograd.grad(loss1, inputs)[0]

inputs_transposed = inputs.permute(0, 2, 1).reshape(batch_size*seq_len, vocab_size)
target_transposed = target.view(batch_size*seq_len)

loss2 = loss(inputs_transposed, target_transposed)
grad2 = torch.autograd.grad(loss2, inputs)[0]

print(torch.allclose(loss1, loss2))
print(torch.allclose(grad1, grad2))
