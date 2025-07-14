import torch
import torch.nn.functional as F
from model.transformer import iLLuMinator
from model.tokenizer import build_tokenizer, encode
from data.prepare import load_data

text = load_data()
stoi, itos = build_tokenizer(text)
vocab_size = len(stoi)
data = torch.tensor(encode(text, stoi), dtype=torch.long)

block_size = 64
batch_size = 32

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

model = iLLuMinator(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1000):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), 'illuminator.pth')
print("Training complete. Model saved as 'illuminator.pth'.")