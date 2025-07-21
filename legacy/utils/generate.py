import torch
from model.transformer import iLLuMinator
from model.tokenizer import build_tokenizer, encode, decode
from data.prepare import load_data

text = load_data()
stoi, itos = build_tokenizer(text)
vocab_size = len(stoi)

model = iLLuMinator(vocab_size)
model.load_state_dict(torch.load('illuminator.pth'))
model.eval()

def generate(start_text, max_new_tokens=100):
    encoded = encode(start_text, stoi)
    if len(encoded) > model.position_embedding.num_embeddings:
        encoded = encoded[-model.position_embedding.num_embeddings:]
    context = torch.tensor([encoded], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(context)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
    return decode(context[0].tolist(), itos)


