from generate import generate

def chat():
    print("iLLuMinator Chatbot (type 'exit' to quit)")
    context = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        context += f"User: {user_input}\nBot: "
        response = generate(context, max_new_tokens=50)
        response_line = response[len(context):].split('\n')[0]
        print("Bot:", response_line)
        context += response_line + "\n"

if __name__ == '__main__':
    chat()