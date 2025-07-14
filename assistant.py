from generate import generate

def ask(prompt):
    system_prompt = (
        "You are a helpful coding assistant.\n"
        "User: " + prompt + "\n"
        "Assistant:"
    )
    response = generate(system_prompt, max_new_tokens=100)
    return response[len(system_prompt):].split('\n')[0]

if __name__ == '__main__':
    print("Code Assistant (type 'exit' to quit)")
    while True:
        query = input("Code> ")
        if query.lower() == "exit":
            break
        answer = ask(query)
        print("Answer:", answer)
