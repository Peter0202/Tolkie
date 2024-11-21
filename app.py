import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

def load_prompt(model_name, prompt_version):
    prompt_folder = "prompts/"
    
    #model-specific-prompt
    prompt_file = f"prompt_v{prompt_version}_{model_name}.txt"
    prompt_path = os.path.join(prompt_folder, prompt_file)
    print(f"Looking for model-specific prompt file at: {prompt_path}")

    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as file:
            return file.read().strip()
    
    #failsafe > unversal
    universal_prompt_file = f"prompt_v{prompt_version}_universal.txt"
    universal_prompt_path = os.path.join(prompt_folder, universal_prompt_file)
    print(f"Model-specific prompt not found. Falling back to universal prompt at: {universal_prompt_path}")

    if os.path.exists(universal_prompt_path):
        with open(universal_prompt_path, "r") as file:
            return file.read().strip()
    
    #no universal either
    print(f"Error: No prompt found for model '{model_name}' and version '{prompt_version}'")
    return None


def send_prompt_to_model(prompt_text, model="fietje"):
    model_url = f"http://localhost:8003/generate" if model == "fietje" else "http://localhost:8002/generate"
    payload = {"prompt": prompt_text}
    
    try:
        response = requests.post(model_url, json=payload)
        response.raise_for_status()
        return response.json().get("response")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to the {model} model.")
        return None

@app.route('/generate', methods=['POST'])
def generate():
    #get model and version from parms
    model_choice = request.json.get("model", "fietje").strip().lower()
    prompt_version = request.json.get("prompt_version", "0")  #0=default

    #validate model choice
    if model_choice not in ["fietje", "geitje"]:
        return jsonify({"error": "Invalid model choice."}), 400

    #load the correct prompt
    default_prompt = load_prompt(model_choice, prompt_version)
    if not default_prompt:
        return jsonify({"error": f"No prompt found for model '{model_choice}' and version '{prompt_version}'."}), 404

    #get custom text
    custom_text = request.json.get("text", "")
    
    #combine prompt with text
    combined_prompt = f"{default_prompt} Translate the following text: {custom_text}"
    
    # Send the combined prompt to the specified model
    translated_text = send_prompt_to_model(combined_prompt, model=model_choice)
    if translated_text:
        return jsonify({"response": translated_text})
    else:
        return jsonify({"error": "Failed to get a response from the model."}), 2000

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
