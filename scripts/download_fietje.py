from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "BramVanroy/fietje-2-instruct"
save_path = "./models/fietje/model"

# ensure directory exists
os.makedirs(save_path, exist_ok=True)

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Saving {model_name} tokenizer and model...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"{model_name} successfully downloaded and saved to {save_path}")
