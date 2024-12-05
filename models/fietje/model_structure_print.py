from transformers import AutoModelForCausalLM

model_name = "./model"
model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)
