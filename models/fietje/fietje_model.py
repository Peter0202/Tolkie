from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
model_name = "BramVanroy/fietje-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    generated_text = text_generator(prompt, max_length=200, return_full_text=False)[0]["generated_text"]
    return jsonify({"response": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
