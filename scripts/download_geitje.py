from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "Rijgersberg/GEITje-7B"
save_path = "./models/geitje/model"

# Ensure the save directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

try:
    print(f"Downloading {model_name}...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Saving {model_name} tokenizer and model...")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print(f"{model_name} successfully downloaded and saved to {save_path}")
except Exception as e:
    print(f"Error downloading or saving {model_name}: {e}")
