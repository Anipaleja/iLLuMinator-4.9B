from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Anipal/iLLuMinator"  # adjust if different path

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# Simple generation
prompt = "Write a Python function that checks whether a string is a palindrome."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

gen = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(gen[0], skip_special_tokens=True))
