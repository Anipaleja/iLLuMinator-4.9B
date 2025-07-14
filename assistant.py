from generate import generate
import re
import requests
from bs4 import BeautifulSoup

def get_live_data(query):
    headers = {'User-Agent': 'Mozilla/5.0'}
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return "Sorry, I couldn't retrieve live data."

    soup = BeautifulSoup(response.text, 'html.parser')
    snippet = soup.find('div', class_='BNeawe').text
    return snippet if snippet else "No useful information found."

def ask(prompt):
    if any(keyword in prompt.lower() for keyword in ["who is", "what is", "latest", "how to", "where is"]):
        return get_live_data(prompt)
    else:
        system_prompt = f"You are a helpful assistant.\nUser: {prompt}\nAssistant:"
        output = generate(system_prompt, max_new_tokens=100)
        return output[len(system_prompt):].split('\n')[0]

if __name__ == '__main__':
    print("iLLuMinator Assistant (type 'exit' to quit)")
    while True:
        query = input("🔎 Ask> ")
        if query.lower() == 'exit':
            break
        answer = ask(query)
        print("💡", answer)