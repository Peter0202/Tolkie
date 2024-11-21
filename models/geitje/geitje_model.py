from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
model_name = "Rijgersberg/GEITje-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided in the request."}), 400
    
    prompt = data["prompt"]
    
    try:
        generated_text = text_generator(prompt, max_length=200, return_full_text=False)[0]["generated_text"]
        return jsonify({"response": generated_text})
    except Exception as e:
        print(f"Error generating text: {e}")
        return jsonify({"error": "Failed to generate text."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)
