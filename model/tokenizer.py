def build_tokenizer(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(indices, itos):
    return ''.join([itos[i] for i in indices])