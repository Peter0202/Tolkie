from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

# Model details
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
save_path = "./models/mistral/model"

# Ensure directory exists
os.makedirs(save_path, exist_ok=True)

# Log in to Hugging Face Hub for gated model access
print("Please log in to Hugging Face (necessary for gated models).")
login()

print(f"Downloading {model_name} tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Downloading {model_name} model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Saving {model_name} tokenizer and model...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"{model_name} successfully downloaded and saved to {save_path}")
