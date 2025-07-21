def load_data(path='input.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
